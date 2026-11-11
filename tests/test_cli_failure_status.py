import pytest
from dex_frax import cli


@pytest.mark.parametrize('status,code',[('failed',1),('budget_exhausted',1),('input_error',2)])
def test_failed_public_conversion_has_nonzero_exit(monkeypatch,tmp_path,status,code):
    monkeypatch.setattr('sys.argv',['dex-frax','convert',str(tmp_path/'source.pt'),str(tmp_path/'out'),'--source','smplx'])
    monkeypatch.setattr(cli,'single_auto',lambda args:dict(status=status))
    with pytest.raises(SystemExit) as error:cli.main()
    assert error.value.code==code


@pytest.mark.parametrize('status',['review','accepted','complete'])
def test_completed_preview_or_accepted_conversion_is_not_an_execution_failure(monkeypatch,tmp_path,status):
    monkeypatch.setattr('sys.argv',['dex-frax','convert',str(tmp_path/'source.pt'),str(tmp_path/'out'),'--source','smplx'])
    monkeypatch.setattr(cli,'single_auto',lambda args:dict(status=status))
    cli.main()


@pytest.mark.parametrize('batch,case,code', [
    ('resource_exhausted', 'resource_exhausted', 1),
    ('complete', 'input_error', 2),
    ('complete', 'failed', 1),
])
def test_public_batch_returns_failure(monkeypatch, tmp_path, batch, case, code):
    from dex_frax import managed_batch, resources

    monkeypatch.setattr('sys.argv', ['dex-frax', 'batch', 'manifest.json', str(tmp_path)])
    monkeypatch.setattr(resources, 'contain_linux', lambda args: None)
    cli.atomic_json(tmp_path / 'batch.json', dict(status=batch))
    monkeypatch.setattr(managed_batch, 'run', lambda args: [dict(status=case)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == code


def test_contained_batch_preserves_child_exit_status(monkeypatch):
    from dex_frax import resources

    monkeypatch.setattr('sys.argv', ['dex-frax', 'batch', 'manifest.json', 'out'])
    monkeypatch.setattr(resources, 'contain_linux', lambda args: 2)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
