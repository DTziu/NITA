# pcapng_parser_fsnet.py - 精简版
import csv
import logging
import os
import struct
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _ether_type(data, offset=0):
    if offset + 14 > len(data):
        return None, offset
    eth = struct.unpack_from('>H', data, offset + 12)[0]
    if eth == 0x8100 and offset + 18 <= len(data):
        return struct.unpack_from('>H', data, offset + 16)[0], offset + 18
    if eth == 0x88a8 and offset + 22 <= len(data):
        return struct.unpack_from('>H', data, offset + 20)[0], offset + 22
    return eth, offset + 14


def _ipv4(data, offset):
    if offset + 20 > len(data) or data[offset] >> 4 != 4:
        return None
    ihl = (data[offset] & 0x0F) * 4
    if ihl < 20 or offset + ihl > len(data):
        return None
    return {
        'protocol': data[offset + 9],
        'src_ip': '.'.join(str(b) for b in data[offset + 12:offset + 16]),
        'dst_ip': '.'.join(str(b) for b in data[offset + 16:offset + 20]),
        'header_length': ihl,
        'total_length': struct.unpack_from('>H', data, offset + 2)[0],
        'version': 4,
    }


def _ipv6(data, offset):
    if offset + 40 > len(data) or data[offset] >> 4 != 6:
        return None
    payload_length = struct.unpack_from('>H', data, offset + 4)[0]
    return {
        'protocol': data[offset + 6],
        'src_ip': ':'.join(f'{data[offset + 8 + i * 2]:02x}{data[offset + 9 + i * 2]:02x}' for i in range(8)),
        'dst_ip': ':'.join(f'{data[offset + 24 + i * 2]:02x}{data[offset + 25 + i * 2]:02x}' for i in range(8)),
        'header_length': 40,
        'total_length': payload_length + 40,
        'version': 6,
    }


def _transport(data, offset, protocol):
    if offset + 8 > len(data):
        return None
    src_port, dst_port = struct.unpack_from('>HH', data, offset)
    if protocol == 6:
        if offset + 20 > len(data):
            return None
        header_len = (data[offset + 12] >> 4) * 4
        if header_len < 20 or offset + header_len > len(data):
            return None
        total = len(data) - offset
        return {
            'src_port': src_port,
            'dst_port': dst_port,
            'header_length': header_len,
            'protocol': 'TCP',
            'transport_length': total,
            'transport_payload_length': max(0, total - header_len),
        }
    if protocol == 17:
        udp_length = struct.unpack_from('>H', data, offset + 4)[0]
        if udp_length < 8:
            return None
        return {
            'src_port': src_port,
            'dst_port': dst_port,
            'header_length': 8,
            'protocol': 'UDP',
            'transport_length': udp_length,
            'transport_payload_length': udp_length - 8,
        }
    return None


def _tls(data, offset):
    if offset + 5 > len(data):
        return {}
    try:
        content_type = data[offset]
        version_major, version_minor = data[offset + 1], data[offset + 2]
        record_length = struct.unpack_from('>H', data, offset + 3)[0]
        tls_info = {
            'tls_record_type': content_type,
            'tls_record_type_name': {
                20: 'change_cipher_spec',
                21: 'alert',
                22: 'handshake',
                23: 'application_data',
                24: 'heartbeat',
            }.get(content_type, f'unknown_{content_type}'),
            'tls_version': f'{version_major}.{version_minor}',
            'tls_record_length': record_length,
        }
        if content_type == 22 and offset + 9 <= len(data):
            handshake_type = data[offset + 5]
            tls_info.update({
                'tls_handshake_type': handshake_type,
                'tls_handshake_name': {
                    0: 'hello_request',
                    1: 'client_hello',
                    2: 'server_hello',
                    4: 'new_session_ticket',
                    11: 'certificate',
                    12: 'server_key_exchange',
                    13: 'certificate_request',
                    14: 'server_hello_done',
                    15: 'certificate_verify',
                    16: 'client_key_exchange',
                    20: 'finished',
                }.get(handshake_type, f'unknown_{handshake_type}'),
            })
        return tls_info
    except Exception as e:
        logger.warning('TLS parsing error at offset %s: %s', offset, e)
        return {}


def _payload(data, offset):
    if offset >= len(data):
        return {}
    payload = {'payload_length': len(data) - offset}
    tls_info = _tls(data, offset) if payload['payload_length'] >= 5 else {}
    if tls_info:
        payload['tls_info'] = tls_info
    return payload


def _read_exact(f, size):
    data = f.read(size)
    return data if len(data) == size else b''


def parse_pcapng_packets(pcapng_path):
    packets, interface_info = [], {}
    stats = {'total_blocks': 0, 'epb_blocks': 0, 'parsed_packets': 0, 'errors': 0}
    try:
        with open(pcapng_path, 'rb') as f:
            if f.read(4) != b'\x0a\x0d\x0d\x0a':
                raise ValueError('无效的 pcapng 文件头')
            shb_len = struct.unpack_from('<I', f.read(4))[0]
            f.seek(shb_len - 8, 1)
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                block_type, block_len = struct.unpack_from('<II', header)
                stats['total_blocks'] += 1
                if block_len < 12:
                    logger.warning('块长度太小: %s', block_len)
                    break
                body = _read_exact(f, block_len - 8)
                if not body:
                    break
                if block_type == 1:
                    if len(body) < 8:
                        continue
                    interface_info[len(interface_info)] = {
                        'linktype': struct.unpack_from('<H', body, 0)[0],
                        'snaplen': struct.unpack_from('<I', body, 4)[0],
                    }
                    continue
                if block_type != 6:
                    continue
                stats['epb_blocks'] += 1
                if len(body) < 20:
                    continue
                interface_id, ts_hi, ts_lo, cap_len, orig_len = struct.unpack_from('<IIIII', body, 0)
                if interface_id not in interface_info:
                    continue
                packet_data = body[20:20 + cap_len]
                if len(packet_data) < 14:
                    continue
                eth_type, eth_offset = _ether_type(packet_data)
                if eth_type not in (0x0800, 0x86dd):
                    continue
                ip_info = _ipv4(packet_data, eth_offset) if eth_type == 0x0800 else _ipv6(packet_data, eth_offset)
                if not ip_info or ip_info['protocol'] not in (6, 17):
                    continue
                transport_offset = eth_offset + ip_info['header_length']
                transport_info = _transport(packet_data, transport_offset, ip_info['protocol'])
                if not transport_info:
                    continue
                payload_info = _payload(packet_data, transport_offset + transport_info['header_length'])
                info = {
                    'timestamp': (ts_hi << 32) | ts_lo,
                    'captured_length': cap_len,
                    'original_length': orig_len,
                    'ip_version': ip_info['version'],
                    'src_ip': ip_info['src_ip'],
                    'dst_ip': ip_info['dst_ip'],
                    'protocol': ip_info['protocol'],
                    'protocol_name': 'TCP' if ip_info['protocol'] == 6 else 'UDP',
                    'src_port': transport_info['src_port'],
                    'dst_port': transport_info['dst_port'],
                    'ip_header_length': ip_info['header_length'],
                    'transport_header_length': transport_info['header_length'],
                    'transport_total_length': transport_info['transport_length'],
                    'transport_payload_length': transport_info['transport_payload_length'],
                    'payload_length': payload_info.get('payload_length', 0),
                    'total_length': ip_info['total_length'],
                }
                if 'tls_info' in payload_info:
                    info.update(payload_info['tls_info'])
                packets.append(info)
                stats['parsed_packets'] += 1
    except Exception as e:
        logger.error('解析文件 %s 时发生错误: %s', pcapng_path, e)
    logger.info('文件 %s 解析完成: %s', os.path.basename(pcapng_path), stats)
    return packets


def extract_tcp_flows_from_pcapng(pcapng_path):
    flows = defaultdict(list)
    for pkt in parse_pcapng_packets(pcapng_path):
        if pkt['protocol'] != 6:
            continue
        forward = (pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'])
        reverse = (pkt['dst_ip'], pkt['src_ip'], pkt['dst_port'], pkt['src_port'])
        key = min(forward, reverse)
        pkt_copy = pkt.copy()
        pkt_copy['direction'] = 1 if key == forward else -1
        flows[key].append(pkt_copy)
    for packets in flows.values():
        packets.sort(key=lambda x: x['timestamp'])
    return flows


def generate_fsnet_features(flow_packets, max_packets=1000):
    lengths = [abs(pkt['captured_length']) for pkt in flow_packets[:max_packets]]
    return lengths + [0] * (max_packets - len(lengths))


def generate_dataset(root_dir='traffic-dataset', output_csv='dataset.csv', max_packets_per_flow=200):
    rows = []
    apps = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    for app in apps:
        app_dir = os.path.join(root_dir, app)
        for pcap_file in sorted(f for f in os.listdir(app_dir) if f.endswith('.pcapng')):
            flows = extract_tcp_flows_from_pcapng(os.path.join(app_dir, pcap_file))
            valid_flows = [packets for packets in flows.values() if len(packets) >= 2]
            for flow_idx, packets in enumerate(valid_flows):
                rows.append([app, pcap_file, flow_idx] + generate_fsnet_features(packets, max_packets_per_flow))
    header = ['所属应用', '所属流量文件名', '流号'] + [f'特征{i + 1}' for i in range(max_packets_per_flow)]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info('生成 %s: %s 条记录', output_csv, len(rows))
    return len(rows)


if __name__ == '__main__':
    generate_dataset()
