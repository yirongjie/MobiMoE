import cv2
import socket
import sys
import threading
import json
import statistics
from psutil import _common as common
import time
import pynvml
pynvml.nvmlInit()
 
class Timer: 
    def __init__(self, name = '', is_verbose = False):
        self._name = name 
        self._is_verbose = is_verbose
        self._is_paused = False 
        self._start_time = None 
        self._accumulated = 0 
        self._elapsed = 0         
        self.start()
 
    def start(self):
        self._accumulated = 0         
        self._start_time = cv2.getTickCount()
 
    def pause(self): 
        now_time = cv2.getTickCount()
        self._accumulated += (now_time - self._start_time)/cv2.getTickFrequency() 
        self._is_paused = True   
 
    def resume(self): 
        if self._is_paused: # considered only if paused 
            self._start_time = cv2.getTickCount()
            self._is_paused = False                      
 
    def elapsed(self):
        if self._is_paused:
            self._elapsed = self._accumulated
        else:
            now = cv2.getTickCount()
            self._elapsed = self._accumulated + (now - self._start_time)/cv2.getTickFrequency()        
        if self._is_verbose is True:      
            name =  self._name
            if self._is_paused:
                name += ' [paused]'
            message = 'Timer::' + name + ' - elapsed: ' + str(self._elapsed) 
            timer_print(message)
        return self._elapsed   
 
class PowerUsage:
    '''
    demo:
        power_usage = PowerUsage()
        power_usage.analyze_start()
        time.sleep(2)
        time_used, power_usage_gpu, power_usage_cpu = power_usage.analyze_end()
        print(time_used)
        print(power_usage_gpu)
        print(power_usage_cpu)
    '''
    def __init__(self):
        self.start_analyze = False
        self.power_usage_gpu_values = list()
        self.power_usage_cpu_values = list()
        self.thread = None
        self.timer = Timer(name='GpuPowerUsage', is_verbose=False)
 
    def analyze_start(self, gpu_id=0, delay=0.1):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        def start():
            self.power_usage_gpu_values.clear()
            self.power_usage_cpu_values.clear()
            self.start_analyze = True
            self.timer.start()
            while self.start_analyze:
                powerusage = pynvml.nvmlDeviceGetPowerUsage(handle)
                self.power_usage_gpu_values.append(powerusage/1000)
 
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                host = socket.gethostname()
                port = 8888
                s.connect((host, port))
                msg = s.recv(1024)
                s.close()
                self.power_usage_cpu_values.append(float(msg.decode('utf-8')))
 
                time.sleep(delay)
        self.thread = threading.Thread(target=start, daemon=True)
        self.thread.start()
 
    def analyze_end(self, mean=True):
        self.start_analyze = False
        while self.thread and self.thread.is_alive():
            time.sleep(0.01)
        time_used = self.timer.elapsed()
        self.thread = None
        power_usage_gpu = statistics.mean(self.power_usage_gpu_values) if mean else self.power_usage_gpu_values
        power_usage_cpu = statistics.mean(self.power_usage_cpu_values) if mean else self.power_usage_cpu_values
        return time_used, power_usage_gpu, power_usage_cpu
 
 
power_usage = PowerUsage()
def power_usage_api(func, note=''):
    @wraps(func)
    def wrapper(*args, **kwargs):
        power_usage.analyze_start()
        result = func(*args, **kwargs)
        print(f'{note}{power_usage.analyze_end()}')
        return result
    return wrapper
 
def power_usage_api2(note=''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            power_usage.analyze_start()
            result = func(*args, **kwargs)
            print(f'{note}{power_usage.analyze_end()}')
            return result
        return wrapper
    return decorator