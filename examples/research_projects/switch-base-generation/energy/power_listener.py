from s_tui.sources.rapl_power_source import RaplPowerSource
import socket
import json
 
def output_to_terminal(source):
    results = {}
    if source.get_is_available():
        source.update()
        source_name = source.get_source_name()
        results[source_name] = source.get_sensors_summary()
    for key, value in results.items():
        print(str(key) + ": ")
        for skey, svalue in value.items():
            print(str(skey) + ": " + str(svalue) + ", ")
 


source = RaplPowerSource()
# output_to_terminal(source)
if not source.get_is_available():
    print("[ERROR] No power source available!")
    exit()
 
s = socket.socket()
host = socket.gethostname()
port = 8888
s.bind((host, port))
s.listen(5)
print("等待客户端连接...")
while True:
    c, addr = s.accept()
    source.update()
    summary = dict(source.get_sensors_summary())
    #msg = json.dumps(summary)
    # package表示CPU，dram表示内存(一般不准)
    power_total = str(sum(list(map(float, [summary[key] for key in summary.keys() if key.startswith('package')]))))
    print(f'发送给{addr}：{power_total}')
    c.send(power_total.encode('utf-8'))
    c.close()                # 关闭连接