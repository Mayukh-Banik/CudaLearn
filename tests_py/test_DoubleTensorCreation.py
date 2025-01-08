import DoubleTensor as dt
import pytest

class TestMyClass:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.a = dt.DoubleTensor(0)
        
    def test_init_correct(self):
        assert self.a.get_index(0) == 0
        assert self.a.shape == ()
        assert self.a.strides == ()
        assert self.a.device == "cuda:0"