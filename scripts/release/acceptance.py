"""Validate four setups in an isolated checkout using a prepared Isaac runtime.

Run this driver outside the directory hidden by --hide-workspace. Bubblewrap
denies access to original source/data; Python dependencies remain available.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--hide-workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    code, output = a.checkout.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(code/'source'), POISE_ROOT=str(code), PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for key in ('SHARPA_WAVE_URDF_PATH', 'AP2AP_ROOT'):
        env.pop(key, None)
    prefix = ['bwrap', '--ro-bind', '/', '/', '--bind', str(code.parent), str(code.parent),
              '--tmpfs', str(a.hide_workspace.resolve()), '--proc', '/proc', '--dev-bind', '/dev', '/dev',
              '--bind', '/tmp', '/tmp', '--chdir', str(code), str(a.python.resolve())]
    # Do not resolve a venv executable to the base interpreter: keep venv identity.
    prefix[-1] = str(a.python.absolute())
    def run(name, command, timeout=240):
        started = time.monotonic()
        with (output / (name + '.log')).open('w') as log:
            r = subprocess.run(prefix + command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f'{name} failed; see {output / (name + ".log")}')
        print(name, 'passed', round(time.monotonic()-started, 1), flush=True)
    run('isolation', ['-c', 'import pathlib,poise; assert not pathlib.Path('+repr(str(a.hide_workspace/'Poise'))+').exists(); print(poise.__file__)'])
    run('tests', ['-m', 'pytest', '-q'], timeout=120)
    report = dict(source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
                  original_workspace_hidden=True, runtime='prepared Isaac Lab; isolated venv shares installed dependencies', setups={})
    cli = ['-m', 'poise.cli']
    for path in sorted((code/'configs/setups').glob('*.json')):
        s = json.loads(path.read_text()); sid=s['id']
        run(sid+'-fetch', cli+['fetch-assets','--setup',sid,'--source',str(a.assets),'--with-reference'])
        run(sid+'-assets', cli+['check-assets','--setup',sid,'--with-reference','--deep'])
        play, train, resume = (output/(sid+'-'+suffix) for suffix in ('rollout','train','resume'))
        run(sid+'-rollout',cli+['rollout','--setup',sid,'--headless','--steps','64','--output',str(play)])
        run(sid+'-train',cli+['train','--setup',sid,'--profile','smoke','--output',str(train)])
        run(sid+'-resume',cli+['train','--setup',sid,'--profile','smoke','--resume',str(train/'model_1.pt'),'--output',str(resume)])
        run(sid+'-verify', ['-c', '''import torch,json
from pathlib import Path
from poise.setups import load_setup,check_checkpoint
a=torch.load(Path('''+repr(str(train))+''')/'model_1.pt',map_location='cpu',weights_only=False)
b=torch.load(Path('''+repr(str(resume))+''')/'model_3.pt',map_location='cpu',weights_only=False)
assert a['infos']['poise']['next_iteration']==2 and b['infos']['poise']['next_iteration']==4
assert b['infos']['poise']['tot_timesteps'] > a['infos']['poise']['tot_timesteps']
assert b['infos']['poise']['rank_states'][0]['common_step_counter'] > a['infos']['poise']['rank_states'][0]['common_step_counter']
assert all(torch.isfinite(v).all() for v in b['model_state_dict'].values())
print(json.dumps(check_checkpoint(load_setup('''+repr(sid)+'''),Path('''+repr(str(resume))+''')/'model_3.pt')))
'''])
        report['setups'][sid] = dict(reference_rollout=json.loads((play/'result.json').read_text()),
                                    training_updates=2, resumed_next_iteration=4, verified=True)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=True
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
