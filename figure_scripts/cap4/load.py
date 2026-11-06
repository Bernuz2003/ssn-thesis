import json, csv, numpy as np
ROOT='/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/c0_e0_vc'
DS=[('dvslip',100,'DVS-Lip'),('dvsgesture',11,'DVS128 Gesture'),('cifar10dvs',10,'CIFAR10-DVS'),('thu_eact_50_chl',50,'THU-EACT-50-CHL')]
def macro_f1(pred,tgt,C):
    f=[]
    for c in range(C):
        tp=np.sum((pred==c)&(tgt==c)); fp=np.sum((pred==c)&(tgt!=c)); fn=np.sum((pred!=c)&(tgt==c))
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
        f.append(2*p*r/(p+r) if p+r else 0.)
    return float(np.mean(f))
def load(d,e):
    s=json.load(open(f'{ROOT}/{d}/{e}/clean_summary.json'))
    rows=list(csv.DictReader(open(f'{ROOT}/{d}/{e}/clean_samples.csv')))
    z=np.load(f'{ROOT}/{d}/{e}/clean.npz')
    ids=[r['sample_id'] for r in rows]; assert ids==list(z['sample_ids'])
    col=lambda k: np.array([float(r[k]) for r in rows])
    out=dict(summary=s, ids=ids, logits=z['logits'], targets=z['targets'], pred=z['logits'].argmax(1),
             duration=col('duration_us'), steps=col('steps'), last=col('last_event_steps'),
             sops=col('full_sops'), macs=col('full_dense_macs'), nt_sops=col('no_tail_sops'), nt_macs=col('no_tail_dense_macs'),
             prefix_logits=z['prefix_logits'])
    if 'no_tail_logits' in z.files: out['nt_pred']=z['no_tail_logits'].argmax(1)
    out['correct']=out['pred']==out['targets']
    return out
