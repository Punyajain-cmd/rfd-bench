from sim import *
sc=dict(bh=.025,bx=.06,gx=.35,mass=.2,mu=.6,kp=200,fmax=6,tol=.03,tp=1.0)
print('clean',run(sc,{})[1:3])
for f in FAULTS:
  for mag in (0.25,1):
    r,s,e,_=run(sc,{f:mag}); print(f,mag,s,round(r[-1]['block_x'],3),round(r[-1]['block_y'],3),round(r[-1]['block_yaw'],2),max(x['contact_unexpected'] for x in r))
