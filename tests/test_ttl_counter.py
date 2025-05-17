from video_sampler.ttl_counter import TTLCounter


def test_expire_one_empty():
    counter = TTLCounter(max_ttl=3)
    assert counter.expire_one() is None


def test_expire_after_tick():
    counter = TTLCounter(max_ttl=2)
    counter.add_item("a")
    counter.tick()
    assert counter.expire_one() is None
    counter.tick()
    assert counter.expire_one() == "a"


def test_expire_all():
    counter = TTLCounter(max_ttl=1)
    counter.add_item("a")
    counter.add_item("b")
    assert list(counter.expire_all()) == ["a", "b"]
    assert len(counter) == 0
