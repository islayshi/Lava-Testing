import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

if __name__ == '__main__':
    # Create a single LIF neuron with:
    # - threshold (vth) = 10
    # - constant bias to force spiking
    lif = LIF(shape=(2,), vth=10., bias_mant=15.0, du=0.0, dv=0.0)

    # Monitor spike output for 5 steps
    monitor = Monitor()
    monitor.probe(target=lif.s_out, num_steps=5)

    # Run simulation
    lif.run(condition=RunSteps(num_steps=5), run_cfg=Loihi1SimCfg())

    # Get and print monitored spikes
    spikes = monitor.get_data()
    print("Spikes recorded:", spikes)

    # Stop processes
    lif.stop()
    monitor.stop()
