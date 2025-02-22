import os
import pippy
from torch.distributed import rpc
from torch import nn

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
HOST  = os.environ["HOST"]
PORT  = os.environ["PORT"]
print(f"My rank is {RANK}")


# first thing to do is to init RCP
print("Waiting for all the nodes...")
rpc.init_rpc(
    f"worker{RANK}", # just an identifier
    rank=RANK,
    world_size=WORLD,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=10, # seconds
        init_method=f"tcp://{HOST}:{PORT}", # head node's address and port
    )
)

# split the model, each process materializes its pipeline stage
driver, stage = pippy.all_compile(
    net,
    num_ranks=WORLD,
    num_chunks=WORLD, # microbatching
    schedule="FillDrain", # feed chunks through the pipeline sequentially
    split_policy=pippy.split_into_equal_size(WORLD), # split the model into specified number of equal-size stages
)
print(stage)

if rank == 0:
    x = torch.randn(4, 128)
    y = driver(x) # only rank 0 is able the call the pipeline's driver
    print(y)

rpc.shutdown()
print("Bye!")