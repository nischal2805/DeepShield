"""
Standalone training monitor — NO LLM, runs locally, costs nothing.

Usage:
    python watch_train.py                      # watches the WCLIP log
    python watch_train.py D:/deepshield_data/train_sbi.log

Refreshes every 15s. Shows: latest epoch, val_auc, best so far, trend arrow,
and flags errors / OOM / completion. Ctrl-C to quit.
"""
import os
import re
import sys
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else 'D:/deepshield_data/train_wclip.log'
INTERVAL = 15

EPOCH_RE = re.compile(r'Epoch\s+(\d+)/(\d+).*?(?:val_auc|auc)[=\s]+([0-9.]+)', re.I)
BEST_RE  = re.compile(r'best auc[:=]\s*([0-9.]+)', re.I)
ERR_RE   = re.compile(r'(traceback|error|out of memory|cuda error|exception)', re.I)
DONE_RE  = re.compile(r'(training complete|done\. best)', re.I)


def read(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().replace('\r', '\n')
    except FileNotFoundError:
        return None


def main():
    print(f'Watching {LOG}  (every {INTERVAL}s, Ctrl-C to stop)\n')
    last_epoch = -1
    aucs = []
    while True:
        txt = read(LOG)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'=== {time.strftime("%H:%M:%S")}  {os.path.basename(LOG)} ===')
        if txt is None:
            print('log not found yet...')
            time.sleep(INTERVAL); continue

        epochs = EPOCH_RE.findall(txt)
        best = BEST_RE.findall(txt)
        if epochs:
            ep, tot, auc = epochs[-1]
            auc = float(auc)
            aucs.append(auc)
            # last 8 epochs
            print(f'\nlatest: epoch {ep}/{tot}   val_auc = {auc:.4f}')
            if best:
                print(f'best so far: {float(best[-1]):.4f}')
            print('\nrecent epochs:')
            for e, t, a in epochs[-8:]:
                bar = '#' * int(float(a) * 40)
                print(f'  ep {e:>3}/{t}  auc {float(a):.4f}  {bar}')
            # trend
            if len(epochs) >= 3:
                d = float(epochs[-1][2]) - float(epochs[-3][2])
                arrow = 'UP' if d > 0.003 else ('DOWN' if d < -0.003 else 'flat')
                print(f'\ntrend (last 2 ep): {arrow} ({d:+.4f})')
        else:
            tail = '\n'.join(l for l in txt.split('\n') if l.strip())[-400:]
            print('\nno epoch lines yet. tail:\n' + tail)

        if DONE_RE.search(txt):
            print('\n*** TRAINING COMPLETE ***')
            if best:
                print(f'final best AUC: {float(best[-1]):.4f}')
            break
        if ERR_RE.search(txt) and not epochs:
            print('\n!!! ERROR DETECTED — check the log !!!')
            tailerr = '\n'.join(l for l in txt.split('\n') if l.strip())[-600:]
            print(tailerr)
            break
        time.sleep(INTERVAL)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nstopped.')
