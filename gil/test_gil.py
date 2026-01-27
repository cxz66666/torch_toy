import threading
import time

# 分析所有线程，GIL情况由pthread_cond_timedwait猜出来
# py-spy record   -f speedscope -t -i -n -r 1000 python test.py

# 只分析GIL线程，如果所有线程都没有GIL锁的话就空着
# 具体参考 https://blog.hidva.com/2025/03/02/the-vanishing-second/
# py-spy record -g  -f speedscope -t -i -n -r 1000 python test.py
# 任务 1: 红队（计算密集型，抢占 GIL）
def red_cpu_task():
    name = "RED_CPU_TASK"
    print(f"[{name}] Starting...")
    while True:
        # 纯 Python 计算，必须持有 GIL
        count = 0
        for i in range(1000000):
            count += 1
        # 稍微歇一下让主线程能结束它，但在计算过程中它是持有锁的
        if stop_event.is_set():
            break

# 任务 2: 蓝队（计算密集型，抢占 GIL）
def blue_cpu_task():
    name = "BLUE_CPU_TASK"
    print(f"[{name}] Starting...")
    while True:
        # 同样的计算逻辑，但是函数名不同，方便在图里区分
        result = 1
        for i in range(1000000):
            result = result * 1.0000001
        if stop_event.is_set():
            break

# 任务 3: 绿队（睡眠/IO，主动释放 GIL）
def green_io_task():
    name = "GREEN_IO_TASK"
    print(f"[{name}] Starting (Releasing GIL)...")
    while not stop_event.is_set():
        # time.sleep 会释放 GIL，所以在 Speedscope 中你会看到这段时间是空白的
        # 或者如果有其他线程在跑，这里会让出执行权
        time.sleep(0.01) 

if __name__ == "__main__":
    stop_event = threading.Event()

    # 创建线程
    t1 = threading.Thread(target=red_cpu_task)
    t2 = threading.Thread(target=blue_cpu_task)
    t3 = threading.Thread(target=green_io_task)

    # 启动
    t1.start()
    t2.start()
    t3.start()

    print("Main: Threads are fighting for GIL. Running for 10 seconds...")
    
    try:
        # 让它们跑 10 秒
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        print("Main: Stopping threads...")
        stop_event.set()
        t1.join()
        t2.join()
        t3.join()