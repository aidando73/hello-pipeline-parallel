import os
from torch.distributed import rpc
import torch.distributed as dist
from torch import nn
import torch
from torch.distributed.pipelining import pipeline, SplitPoint

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128,   8)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x

net = Net()
net.eval()

RANK  = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD"])
print(f"My rank is {RANK}")


# first thing to do is to init RCP
print("Waiting for all the nodes...")
dist.init_process_group(
    backend="nccl",
    rank=RANK,
    world_size=WORLD,
)


# split the model, each process materializes its pipeline stage
x = torch.randn(4, 128)
pipe = pipeline(
    module=net,
    mb_args=(x,),
    split_spec={
        "fc2": SplitPoint.BEGINNING,
    }
)

print(pipe)
print("num_stages", pipe.num_stages)

print(f"stage {RANK}:")
print(pipe.get_stage_module(0))

stage = pipe.build_stage(RANK, device=torch.device("cuda"))

schedule = ScheduleGPipe(stage)

if RANK == 0:
    schedule.step(x)
else:
    output = schedule.step()
    print("output", output)

rpc.shutdown()
print("Bye!")