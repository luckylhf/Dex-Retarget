from pathlib import Path
import pytest
from dex_frax import resources as r


@pytest.mark.parametrize('contents,expected',[
    ('file 100\ninactive_file 90\nshmem 10\nfile_dirty 15\nfile_writeback 5\n',70),
    ('file 100\ninactive_file 20\nshmem 0\n',20),
    ('file 100\ninactive_file 20\nactive_file 80\nfile_mapped 10\nfile_dirty 15\nfile_writeback 5\n',70),
    ('file 100\ninactive_file 100\nshmem 100\n',0),
    ('file 100\ninactive_file 100\nfile_dirty 80\nfile_writeback 50\n',0),
    ('invalid',0),
])
def test_only_clean_unmapped_file_counts(tmp_path,contents,expected):
    (tmp_path/'memory.stat').write_text(contents)
    assert r.cgroup_clean_file_cache(tmp_path)==expected


def test_missing_stat_is_conservative(tmp_path):
    assert r.cgroup_clean_file_cache(tmp_path)==0


def test_full_hash_cache_does_not_block_but_ancestor_limit_still_applies(tmp_path,monkeypatch):
    child=tmp_path/'child';parent=tmp_path/'parent'
    for folder,maximum,current,cache in [(child,40,39,35),(parent,50,48,0)]:
        folder.mkdir()
        for name,value in [('memory.max',maximum*r.GIB),('memory.current',current*r.GIB)]:
            (folder/name).write_text(str(value))
        (folder/'memory.stat').write_text(f'file {cache*r.GIB}\ninactive_file {cache*r.GIB}\n')
    original=Path.read_text
    def read(path,*a,**kw):
        if str(path)=='/proc/meminfo':return f'MemTotal: {64*1024**2} kB\nMemAvailable: {54*1024**2} kB\n'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(r.sys,'platform','linux')
    monkeypatch.setattr(r,'cgroup_ancestors',lambda:[child])
    s=r.memory()
    assert s['cgroup_headroom']==36*r.GIB and s['cgroup_limit']==40*r.GIB
    assert r.can_admit(s,12*r.GIB,100*r.GIB,80*r.GIB,[],4*r.GIB)[0]
    monkeypatch.setattr(r,'cgroup_ancestors',lambda:[child,parent])
    assert r.memory()['cgroup_headroom']==2*r.GIB
    assert not r.can_admit(r.memory(),12*r.GIB,100*r.GIB,80*r.GIB,[],4*r.GIB)[0]
