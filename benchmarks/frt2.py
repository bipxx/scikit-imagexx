"""Prospective public one-thread cases. No execution authority."""
def case_specs():
 return [dict(id='transform.frt2'+('/n'+str(n)),api='frt2',n=n,dtype='uint32',layout=layout)for n,layout in [(31,'C'),(127,'C'),(127,'F'),(127,'reverse')]]
METHOD=dict(threads=1,repeat=7,target_ms=100,warmup_min_ms=250,retain_all_losses=True,status='UNBOUND_NO_TIMING')
