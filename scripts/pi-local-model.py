"""Serialize local model transitions for Pi. No inference prompts are sent."""
import errno
import fcntl
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME_DIR = Path.home()
SETUP = HOME_DIR / 'Developer/ai/deepseek-v41-local'
MTPLX = HOME_DIR / 'Developer/ai/mtplx'
MTPLX_REPO = 'Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed'
STATE = HOME_DIR / '.local/state/nvim/pi-models'
QWEN = 'qwen3.8:27b-mtp-bf16'
INKLING_MLX = 'Inkling-2-Mini-oQ4e-mtp'
OMLX = HOME_DIR / 'Developer/ai/inkling-mlx'
TARGETS = {'inkling-mlx': (8016, INKLING_MLX), 'ollama': (11434, QWEN),
           'ds4-q4': (8014, 'deepseek-v4.1-flash'),
           'mtplx': (8000, 'mtplx-qwen38-27b-optimized-speed')}


def request(port, path, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 headers={'Content-Type': 'application/json',
                                          **({'Authorization': 'Bearer inkling-local'} if port == 8016 else {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def loaded():
    try:
        return request(11434, '/api/ps')['models']
    except urllib.error.URLError as error:
        if getattr(error.reason, 'errno', None) == errno.ECONNREFUSED:
            return []
        raise


def wait_for(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise RuntimeError(f'Timed out waiting for {description}')


def processes():
    output = subprocess.check_output(['ps', '-axo', 'pid=,stat=,command='], text=True)
    found = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or 'Z' in parts[1]:
            continue
        pid, _, command = parts
        try:
            args = shlex.split(command)
        except ValueError:
            continue
        # oMLX changes its process title after startup, hiding argv.
        if args == ['omlx-server']:
            try:
                managed_pid = int((STATE / 'inkling-mlx.pid').read_text())
            except (OSError, ValueError):
                managed_pid = None
            if managed_pid == int(pid):
                listener = subprocess.run(
                    ['/usr/sbin/lsof', '-a', '-p', pid, '-iTCP:8016', '-sTCP:LISTEN'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if listener.returncode == 0:
                    found[int(pid)] = 'inkling-mlx'
            continue
        if ((args[0] == str(OMLX / '.venv/bin/omlx') or
                 (Path(args[0]).name.startswith('python') and args[1:2] == [str(OMLX / '.venv/bin/omlx')])) and 'serve' in args
                and '--port' in args and args[args.index('--port') + 1:args.index('--port') + 2] == ['8016']
                and '--model-dir' in args and args[args.index('--model-dir') + 1:args.index('--model-dir') + 2] == [str(OMLX / 'models')]):
            found[int(pid)] = 'inkling-mlx'
            continue
        launcher = str(MTPLX / '.venv/bin/mtplx') in args[:2] and 'serve' in args
        # The CLI execs the actual server module and resolves the repo to a local path.
        module_server = ('-m' in args[:4]
                         and args[args.index('-m') + 1:args.index('-m') + 2]
                         == ['mtplx.server.openai'])
        if (launcher or module_server) and '--model' in args and '--port' in args:
            model = args[args.index('--model') + 1:args.index('--model') + 2]
            model_path = HOME_DIR / '.mtplx/models' / MTPLX_REPO.replace('/', '--')
            if (model in ([MTPLX_REPO], [str(model_path)])
                    and args[args.index('--port') + 1:args.index('--port') + 2] == ['8000']):
                found[int(pid)] = 'mtplx'
            continue
        if not args or Path(args[0]).name != 'ds4-server':
            continue
        for provider, (port, _) in TARGETS.items():
            if not provider.startswith('ds4-'):
                continue
            quant = provider[-2:].upper()
            if ('--port' in args and args.index('--port') + 1 < len(args)
                    and args[args.index('--port') + 1] == str(port) and
                    f'DeepSeek-V4.1-Flash-{quant}.gguf' in command):
                found[int(pid)] = provider
    return found


def stop_local_servers(except_provider=None):
    stopping = {pid: provider for pid, provider in processes().items()
                if provider != except_provider}
    for pid in stopping:
        # Recheck identity immediately before sending a signal.
        if processes().get(pid) == stopping[pid]:
            print(f'Stopping {stopping[pid]} (PID {pid})', flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    wait_for(lambda: not (set(processes()) & set(stopping)), 60, 'local model server to exit')


def unload_ollama(keep=None):
    for model in loaded():
        name = model.get('name', model.get('model'))
        if name != keep:
            print(f'Unloading {name}', flush=True)
            request(11434, '/api/generate', {'model': name, 'keep_alive': 0,
                                           'stream': False}, timeout=120)
    wait_for(lambda: all(m.get('name', m.get('model')) == keep for m in loaded()),
             60, 'Ollama to release the previous model')


def ready(provider):
    port, model = TARGETS[provider]
    try:
        if provider == 'mtplx':
            health = request(port, '/health')
            return health.get('ok') is True and health.get('model') == model
        return any(m['id'] == model for m in request(port, '/v1/models')['data'])
    except (OSError, ValueError, KeyError):
        return False


def ensure(provider):
    if provider not in TARGETS:
        raise ValueError(f'Unsupported local provider: {provider}')
    if provider.startswith('ds4-') and not (SETUP / 'start-server.sh').is_file():
        raise RuntimeError('DeepSeek start-server.sh is missing')
    if provider == 'mtplx' and not (MTPLX / '.venv/bin/mtplx').is_file():
        raise RuntimeError('MTPLX executable is missing')
    if provider == 'inkling-mlx' and not (OMLX / '.venv/bin/omlx').is_file():
        raise RuntimeError('oMLX executable is missing')
    stop_local_servers(except_provider=provider)
    ollama_model = TARGETS[provider][1] if provider == 'ollama' else None
    unload_ollama(keep=ollama_model)
    if ollama_model:
        required_ctx = 32768
        if not any(m.get('name') == ollama_model and m.get('context_length', 0) >= required_ctx for m in loaded()):
            print(f'Loading {ollama_model}', flush=True)
            request(11434, '/api/generate', {
                'model': ollama_model, 'stream': False, 'keep_alive': '30m',
                'options': {'num_ctx': required_ctx},
            }, timeout=300)
        if not ready(provider):
            raise RuntimeError(f'{ollama_model} is not available in Ollama')
    else:
        running = provider in processes().values()
        if not running:
            if ready(provider):
                raise RuntimeError('Endpoint is occupied by an unrecognized server; not starting another')
            log = STATE / f'{provider}.log'
            print(f'Starting {provider}; log: {log}', flush=True)
            with log.open('ab') as output:
                command = ([str(OMLX / '.venv/bin/omlx'), 'serve',
                            '--model-dir', str(OMLX / 'models'), '--base-path', str(OMLX / 'state'),
                            '--host', '127.0.0.1', '--port', '8016', '--no-hf-cache',
                            '--max-concurrent-requests', '1', '--paged-ssd-cache-max-size', '20GB',
                            '--api-key', 'inkling-local', '--headless']
                           if provider == 'inkling-mlx' else
                           [str(MTPLX / '.venv/bin/mtplx'), 'serve', '--model', MTPLX_REPO,
                            '--host', '127.0.0.1', '--port', '8000',
                            '--reasoning-effort', 'medium', '--fan-mode', 'default']
                           if provider == 'mtplx' else
                           ['sh', str(SETUP / 'start-server.sh'), provider[-2:]])
                child = subprocess.Popen(command,
                                         stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                         start_new_session=True)
            (STATE / f'{provider}.pid').write_text(str(child.pid))
        else:
            child = None
        def healthy():
            if child is not None and child.poll() is not None:
                raise RuntimeError(f'{provider} exited; see {STATE / (provider + ".log")}')
            return ready(provider)
        try:
            wait_for(healthy, 300, f'{provider} endpoint to become ready')
        except Exception:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=60)
            raise
    print(f'Ready: {provider}', flush=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'transition.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another local model transition is in progress')
        ensure(sys.argv[1])


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
