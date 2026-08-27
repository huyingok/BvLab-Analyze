import os
import subprocess
from os.path import join
import sys
import traceback


def SignMaskToGMM(workQue, finishQue, errorQue, savePath, gpuId, exePath, processDir, proceLs):
    while True:
        try:
            add = workQue.get(timeout=2)  # 设置超时避免阻塞
            if len(add) == 0:
                return
            name = os.path.basename(add)
            if name in proceLs:
                finishQue.put(1)
                continue
            print('%s %s %s %d' % (exePath, add, savePath, gpuId))
            p = subprocess.Popen('%s %s %s %d' % (exePath, add, savePath, gpuId))
            p.wait()
            with open(join(processDir, name), 'w') as f:
                f.write('ok')
            finishQue.put(1)
        except Exception as e:
            if errorQue.qsize() == 0:
                print("\n=== Error message ===")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("=== Error location ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    print(f"  File: {frame.filename}")
                    print(f"  Line number: {frame.lineno}")
                    print(f"  Function: {frame.name}")
                    print(f"  Code: {frame.line}\n")
            errorQue.put(e)


def QueGmmToSwc(nameLsQue, errorQue, exePath, processDir, proceLs):
    while True:
        try:
            add = nameLsQue.get(timeout=1)
            name = os.path.basename(add)
            if name in proceLs:
                continue
            print(add)
            p = subprocess.Popen('%s %s' % (exePath, add))
            p.wait()
            with open(join(processDir, name), 'w') as f:
                f.write('ok')
        except Exception as e:
            if errorQue.qsize() == 0:
                print("\n=== Error message ===")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("=== Error location ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    print(f"  File: {frame.filename}")
                    print(f"  Line number: {frame.lineno}")
                    print(f"  Function: {frame.name}")
                    print(f"  Code: {frame.line}\n")
            errorQue.put(e)

