"""内存看门狗: 只盯**本项目自己**的进程, 超限自动 kill。

★口径: 自己的占用用 PSS(/proc/*/smaps_rollup) 求和 —— 不能用 ps RSS 求和,
  fork 出来的 worker 共享页面, RSS 相加会重复计数(判负库里这个错误报过两次 2.6 倍)。
★系统整体用 free 的"已用"列。
★不用 pgrep -f 找进程: 它会匹配到发出命令的 shell 自己(CLAUDE.md 记了三次)。
  改为遍历 /proc, 且必须 readlink /proc/<pid>/exe 确认是 python, 并排除自己。
"""
import os as _os
import os, sys, time, signal

MARKS = [_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
MY_LIMIT_GB = 300                    # 自己的 PSS 上限
SYS_LIMIT_GB = 1900                  # 系统 free"已用"上限(总 2003G, 给别人留余量)
LOG = '{OUTDIR}/memguard.log'


def my_pids():
    me, parent = os.getpid(), os.getppid()
    out = []
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        pid = int(p)
        if pid in (me, parent):
            continue
        try:
            if os.stat(f'/proc/{p}').st_uid != os.getuid():
                continue
            exe = os.readlink(f'/proc/{p}/exe')
            if 'python' not in os.path.basename(exe):        # ★必须是 python
                continue
            cl = open(f'/proc/{p}/cmdline', 'rb').read().decode('utf8', 'replace')
            if 'memguard' in cl:
                continue
            # ★认 cwd 而不是只认 cmdline —— 相对路径启动时 cmdline 里没有项目绝对路径
            cwd = os.readlink(f'/proc/{p}/cwd')
            if not any(m in cwd or m in cl for m in MARKS):
                continue
            out.append(pid)
        except Exception:
            continue
    return out


def pss_gb(pids):
    tot = 0
    for pid in pids:
        try:
            for ln in open(f'/proc/{pid}/smaps_rollup'):
                if ln.startswith('Pss:'):
                    tot += int(ln.split()[1]); break
        except Exception:
            pass
    return tot / 1048576.0


def sys_used_gb():
    m = {}
    for ln in open('/proc/meminfo'):
        k, v = ln.split(':'); m[k] = int(v.split()[0])
    # 与 free 的"已用"同口径: total - free - buffers - cached - sreclaimable
    return (m['MemTotal'] - m['MemFree'] - m['Buffers'] - m['Cached']
            - m.get('SReclaimable', 0)) / 1048576.0


def log(s):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%H:%M:%S")} {s}\n')


if __name__ == '__main__':
    log(f'看门狗启动 pid={os.getpid()} 自限{MY_LIMIT_GB}G 系统限{SYS_LIMIT_GB}G')
    while True:
        pids = my_pids()
        mine, tot = pss_gb(pids), sys_used_gb()
        if pids and (mine > MY_LIMIT_GB or tot > SYS_LIMIT_GB):
            why = f'自己{mine:.0f}G>{MY_LIMIT_GB}' if mine > MY_LIMIT_GB else f'系统{tot:.0f}G>{SYS_LIMIT_GB}'
            log(f'!!! 超限 {why} —— kill {len(pids)} 个进程 {pids}')
            for pid in pids:
                try: os.kill(pid, signal.SIGTERM)
                except Exception: pass
            time.sleep(5)
            for pid in pids:
                try: os.kill(pid, signal.SIGKILL)
                except Exception: pass
        elif pids:
            log(f'本项目 {len(pids)} 进程 PSS={mine:.1f}G  系统已用={tot:.0f}G')
        time.sleep(30)
