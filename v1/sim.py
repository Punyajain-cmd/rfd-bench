"""MuJoCo planar push task. Pusher (2 slide joints, position actuators) pushes a block to a goal along +x."""
import mujoco, numpy as np
XML = """<mujoco model="push"><option timestep="0.002" integrator="implicitfast"/>
<worldbody><light pos="0 0 2"/><camera name="top" pos="0.25 0 0.9" xyaxes="1 0 0 0 1 0"/>
<geom name="floor" type="plane" size="1 1 .1" rgba=".8 .8 .8 1" friction="{mu} .005 .0001"/>
<site name="goal" pos="{gx} 0 0.001" size="{tol} .001" type="cylinder" rgba="0 1 0 .4"/>
<body name="pusher" pos="0 0 0.03"><joint name="px" type="slide" axis="1 0 0" damping="5"/><joint name="py" type="slide" axis="0 1 0" damping="5"/>
<geom name="pusher" type="box" size=".01 .04 .015" friction=".05 .005 .0001" mass="0.5" rgba=".2 .2 .9 1" contype="1" conaffinity="1"/></body>
<body name="block" pos="{bx} 0 {bh}"><freejoint/><geom name="block" type="box" size="{bh} {bh} {bh}" mass="{mass}" friction="{mu} .005 .0001" rgba=".9 .3 .2 1"/></body>
{obst}
</worldbody>
<actuator><position name="ax" joint="px" kp="{kp}" forcerange="-{fmax} {fmax}" forcelimited="true"/>
<position name="ay" joint="py" kp="{kp}" forcerange="-{fmax} {fmax}" forcelimited="true"/></actuator></mujoco>"""
FAULTS = ["low_friction","heavy_payload","weak_actuator","hidden_obstacle","encoder_bias"]
def build(scene, faults):
    """scene: dict(bh,bx,gx,speed,mass,mu,kp,fmax,tol). faults: dict name->magnitude in [0,1] (1=full)."""
    s=dict(scene); f=faults
    mu=s['mu']*(1-0.93*f.get('low_friction',0)); mass=s['mass']*(1+14*f.get('heavy_payload',0))
    fmax=s['fmax']*(1-0.85*f.get('weak_actuator',0))
    obst=""
    if f.get('hidden_obstacle',0)>0:
        ox=s['bx']+0.5*(s['gx']-s['bx']); h=0.01+0.03*f['hidden_obstacle']
        obst=f'<body name="obst" pos="{ox:.3f} 0 {h/2:.4f}"><geom name="obst" type="box" size=".01 .06 {h/2:.4f}" rgba=".4 .4 .4 1"/></body>'
    xml=XML.format(mu=mu,gx=s['gx'],tol=s['tol'],bx=s['bx'],bh=s['bh'],mass=mass,obst=obst,kp=s['kp'],fmax=fmax)
    return mujoco.MjModel.from_xml_string(xml), dict(mu=mu,mass=mass,fmax=fmax)
def run(scene, faults, n_obs=40, T=4.0, render=None):
    m,eff=build(scene,faults); d=mujoco.MjData(m)
    bias_x=0.05*faults.get('encoder_bias',0)  # encoder reports x + bias
    gid={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n):n for n in ['pusher','block','floor','obst'] if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n)>=0}
    gid={v:k for k,v in gid.items()}; gid={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,k):k for k in gid}
    start=-s_start(scene); end=scene['gx']-scene['bh']-0.011
    steps=int(T/m.opt.timestep); every=steps//n_obs; rows=[]; frames=[]
    d.qpos[0]=start; mujoco.mj_forward(m,d)
    for i in range(steps):
        t=i*m.opt.timestep; frac=min(1,max(0,t-scene.get('delay',0))/scene['tp']); cmdx=start+(end-start)*frac
        enc_x=d.qpos[0]+bias_x; enc_y=d.qpos[1]
        d.ctrl[0]=cmdx-bias_x; d.ctrl[1]=0.0  # position servo closes loop on encoder  # target y=0 in encoder frame
        mujoco.mj_step(m,d)
        if (i+1)%every==0:
            pb=pob=0; 
            for c in d.contact[:d.ncon]:
                a,b=sorted([gid.get(c.geom1,'?'),gid.get(c.geom2,'?')])
                if (a,b)==('block','pusher'): pb=1
                if 'obst' in (a,b) and ('block' in (a,b) or 'pusher' in (a,b)): pob=1
            bq=d.qpos[2:9]; yaw=2*np.arctan2(bq[6],bq[3])
            rows.append(dict(t=len(rows),cmd_x=cmdx,cmd_y=0.0,enc_x=enc_x,enc_y=enc_y,force_x=float(d.actuator_force[0]),
               block_x=float(bq[0]),block_y=float(bq[1]),block_yaw=float(yaw),contact_pusher_block=pb,contact_unexpected=pob,_true_py=float(d.qpos[1])))
            if render is not None and len(rows) in render: frames.append(render_frame(m,d))
    bx,by=rows[-1]['block_x'],rows[-1]['block_y']
    success= ((bx-scene['gx'])**2+by**2)**.5 < scene['tol'] and abs(rows[-1]['block_yaw'])<0.35
    return rows, bool(success), eff, frames
def s_start(scene): return 0.1
_R={}
def render_frame(m,d):
    if id(m) not in _R: _R.clear(); _R[id(m)]=mujoco.Renderer(m,160,240)
    r=_R[id(m)]; r.update_scene(d,camera='top'); return r.render().copy()
