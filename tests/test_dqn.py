"""Shape and logic checks for the from-scratch DQN (skipped without torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from notifyrl.config import AgentConfig  # noqa: E402
from notifyrl.dqn import DQNAgent, QNetwork  # noqa: E402
from notifyrl.env import OBS_DIM, make_env  # noqa: E402
from notifyrl.replay import TransitionMemory  # noqa: E402


@pytest.mark.parametrize("dueling", [False, True])
def test_qnetwork_output_shape(dueling):
    net = QNetwork(OBS_DIM, 4, (64, 64), dueling)
    out = net(torch.zeros(7, OBS_DIM))
    assert out.shape == (7, 4)


def test_dueling_advantages_are_centred():
    torch.manual_seed(0)
    net = QNetwork(OBS_DIM, 4, (32,), dueling=True)
    x = torch.randn(5, OBS_DIM)
    z = net.trunk(x)
    q = net(x)
    v = net.value_head(z)
    assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-5)


def test_replay_sampling_shapes():
    mem = TransitionMemory(100, OBS_DIM, seed=0)
    for i in range(30):
        mem.push(np.full(OBS_DIM, i, np.float32), i % 4, float(i), np.zeros(OBS_DIM), i % 2 == 0)
    b = mem.sample(16)
    assert b["obs"].shape == (16, OBS_DIM)
    assert b["actions"].shape == (16,)
    assert set(np.unique(b["dones"])) <= {0.0, 1.0}


def test_replay_ring_overwrites_oldest():
    mem = TransitionMemory(5, 2, seed=0)
    for i in range(8):
        mem.push(np.array([i, i]), 0, 0.0, np.zeros(2), False)
    assert len(mem) == 5
    assert sorted(mem.obs[:, 0].tolist()) == [3, 4, 5, 6, 7]


@pytest.mark.parametrize("double", [False, True])
def test_td_target_terminal_ignores_bootstrap(double):
    cfg = AgentConfig(hidden=(16,), double=double, dueling=False)
    agent = DQNAgent(OBS_DIM, 4, cfg, seed=0)
    r = torch.tensor([1.5, -2.0])
    nxt = torch.randn(2, OBS_DIM)
    y = agent._td_target(r, nxt, torch.ones(2))
    assert torch.allclose(y, r)


def test_epsilon_schedule_monotone():
    agent = DQNAgent(OBS_DIM, 4, AgentConfig(hidden=(8,)), seed=0, total_steps=1000)
    values = []
    for s in (0, 100, 300, 600, 900):
        agent.steps = s
        values.append(agent.epsilon())
    assert values[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(0.05)


def test_update_reduces_loss_on_fixed_batch():
    cfg = AgentConfig(hidden=(32,), warmup=32, batch_size=32, lr=1e-2)
    agent = DQNAgent(OBS_DIM, 4, cfg, seed=0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(32, OBS_DIM)).astype(np.float32)
    for i in range(32):
        agent.memory.push(obs[i], i % 4, 1.0, obs[i], True)
    first = agent._update()
    for _ in range(50):
        last = agent._update()
    assert last < first


def test_save_load_roundtrip(tmp_path):
    agent = DQNAgent(OBS_DIM, 4, AgentConfig(hidden=(16, 16)), seed=1)
    x = np.random.default_rng(0).normal(size=(3, OBS_DIM)).astype(np.float32)
    path = tmp_path / "a.pt"
    agent.save(str(path), extra={"tag": "t", "frame_stack": 1})
    clone = DQNAgent.load(str(path))
    assert np.allclose(agent.q_values(x), clone.q_values(x))
    assert clone.meta["tag"] == "t"


def test_agent_runs_in_env():
    env = make_env()
    agent = DQNAgent(OBS_DIM, 4, AgentConfig(hidden=(16,), warmup=10, batch_size=8), seed=0, total_steps=500)
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        a = agent.act(obs)
        nxt, r, done, _, _ = env.step(a)
        agent.observe(obs, a, r, nxt, done)
        obs = nxt if not done else env.reset(seed=1)[0]
    assert agent.steps == 40
