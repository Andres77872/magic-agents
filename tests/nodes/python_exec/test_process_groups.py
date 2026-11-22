import asyncio
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from magic_agents.node_system.python_code_runner import CodeRunner
from magic_agents.util.aio import wait_for


pytestmark = pytest.mark.skipif(
    os.name != 'posix' or not Path('/proc/self/stat').exists(),
    reason='POSIX process groups verified using Linux process state',
)


def running(pid):
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    # A terminated orphan may briefly await reaping by init; it cannot run.
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (FileNotFoundError, ProcessLookupError):
        # The process vanished between the signal probe and the /proc read.
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome',['cancel','timeout','success','crash'])
async def test_code_runner_reaps_descendants_for_every_exit_path(tmp_path,outcome):
    pidfile=tmp_path/'descendant.pid'
    code='''def run(handler):
    import os,subprocess,sys,time
    from pathlib import Path
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'])
    Path(handler['pidfile']).write_text(str(child.pid))
    if handler['outcome']=='success': return {'value':14}
    if handler['outcome']=='crash': os._exit(7)
    time.sleep(10)
    return {'unexpected':True}
'''
    runner=CodeRunner(timeout=1.5 if outcome=='timeout' else 10)
    baseline={p.pid for p in multiprocessing.active_children()}
    task=asyncio.create_task(runner.execute(code,{'pidfile':str(pidfile),'outcome':outcome}))
    pid=None
    try:
        for _ in range(200):
            if pidfile.exists():break
            if task.done():pytest.fail(f'worker ended before child started: {task.result()}')
            await asyncio.sleep(.01)
        else:pytest.fail('worker did not write descendant PID')
        pid=int(pidfile.read_text())
        if outcome=='cancel':
            started=time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await wait_for(task,2)
            assert time.monotonic()-started<3
        else:
            result=await wait_for(task,3)
            if outcome=='success':assert result=={'result':{'value':14}}
            else:assert 'error' in result
            if outcome=='timeout':assert 'timed out' in result['error']
        # Poll generously: reaping a killed group is slower on a loaded machine.
        for _ in range(300):
            if not running(pid):break
            await asyncio.sleep(.01)
        assert not running(pid),'The owned descendant survived the tool invocation'
        assert not ({p.pid for p in multiprocessing.active_children()}-baseline)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        if pid is not None and running(pid):
            os.kill(pid,9)


def test_cancellation_before_ready_never_runs_user_code_or_signals_parent_group(tmp_path):
    marker=tmp_path/'must-not-run'
    cancelled=threading.Event()
    cancelled.set()
    parent_group=os.getpgrp()
    baseline={p.pid for p in multiprocessing.active_children()}
    result=CodeRunner(timeout=5)._execute_subprocess_sync(
        'def run(handler):\n    from pathlib import Path\n    Path(handler["marker"]).write_text("ran")',
        {'marker':str(marker)},cancelled)
    assert result=={'error':'execution cancelled'}
    assert not marker.exists()
    assert os.getpgrp()==parent_group
    assert not ({p.pid for p in multiprocessing.active_children()}-baseline)
