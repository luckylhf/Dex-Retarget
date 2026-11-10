import numpy as np
import jax
import jax.numpy as jnp
from dex_frax.v8.robot import Robot
from dex_frax.frax_foot_constraints import FootGeometry


def test_initial_foot_rows_use_signed_distance_jacobian_and_protect_upstream():
    r=Robot();saved=r.d.qpos.copy();constraint=FootGeometry(object(),r,50.)
    np.testing.assert_array_equal(r.d.qpos,saved)
    x=np.r_[0.,0.,1.,np.zeros(29)];ids=np.asarray(sum([g['leg'] for g in r.sides.values()],[]))
    x[ids]=np.random.default_rng(12).uniform(-.2,.2,len(ids));G,h=constraint.measure(x,np.eye(3))
    for j in [r.sides['Left']['leg'][4],r.sides['Right']['leg'][5]]:
        step=np.zeros(32);step[j]=1e-6
        numeric=(constraint.measure(x+step,np.eye(3))[1]-constraint.measure(x-step,np.eye(3))[1])/2e-6
        np.testing.assert_allclose(numeric,-G[:,j],atol=2e-5)
    mask=np.ones(32,bool);mask[ids]=False
    g,b=jax.jit(constraint.rows)(jnp.asarray(x),{'root_r':jnp.eye(3),'priority_mask':jnp.asarray(mask)})
    np.testing.assert_array_equal(np.asarray(g)[:,mask],0.)
    np.testing.assert_allclose(b,50*h)
    assert len(constraint.report['scope'])==constraint.num_rows
