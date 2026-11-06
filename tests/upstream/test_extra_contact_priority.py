"""Regression for uncertain proximity hints competing with a valid foot plant."""
import numpy as np
import pytest
from dex_frax.v8.contact_tasks import body_height_task


def context(support):
    return dict(body_contact_height_tolerance_m=.028,
                sides={'Left':{'contact':np.array([support])},
                       'Right':{'contact':np.array([0.])}})


def test_body_support_is_retained_without_a_planted_foot():
    cfg={'body_contact_loss':'bounded_when_feet_planted'}
    assert body_height_task(.20,context(0.),0,{},cfg)==(.20,1.)


def test_incompatible_extra_hint_cannot_keep_recruiting_waist():
    cfg={'body_contact_loss':'bounded_when_feet_planted'}
    value,slope=body_height_task(.20,context(1.),0,{},cfg)
    assert 0<value<.028
    assert slope<.01
    # An already aligned support retains its local restoring task.
    assert body_height_task(0.,context(1.),0,{},cfg)==(0.,1.)


@pytest.mark.parametrize('support',[0.,.2,1.])
@pytest.mark.parametrize('height',[-.2,-.01,0.,.01,.2])
def test_contact_task_analytic_derivative_and_continuity(height,support):
    cfg={'body_contact_loss':'bounded_when_feet_planted'};t=context(support)
    value,slope=body_height_task(height,t,0,{},cfg);eps=1e-7
    fd=(body_height_task(height+eps,t,0,{},cfg)[0]-body_height_task(height-eps,t,0,{},cfg)[0])/(2*eps)
    assert slope==pytest.approx(fd,abs=2e-9)
    if support<1.:
        delta=1e-5
        other=body_height_task(height,context(support+delta),0,{},cfg)[0]
        assert abs(other-value)<=abs(height)*delta+1e-12
