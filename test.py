import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

if __name__ == '__main__':
    # Create processes
    lif1 = LIF(shape=(3,), vth=10., dv=0.1, du=0.1)
    dense = Dense(weights=np.random.rand(2, 3))
    lif2 = LIF(shape=(2,), vth=10., dv=0.1, du=0.1)

    # Connect processes
    lif1.s_out.connect(dense.s_in)
    dense.a_out.connect(lif2.a_in)

    # Monitor the output of lif2
    monitor = Monitor()
    monitor.probe(target=lif2.s_out, num_steps=5)  # <-- probe s_out for 5 steps

    # Run simulation
    lif2.run(condition=RunSteps(num_steps=5), run_cfg=Loihi1SimCfg())

    # Get monitored spikes
    spikes = monitor.get_data()
    print("LIF2 spikes:", spikes)

    # Stop processes
    lif2.stop()
    dense.stop()
    lif1.stop()
    monitor.stop()
