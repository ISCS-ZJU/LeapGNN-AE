import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

""" Implementation of a ring-reduce with addition. """
# def allreduce(send, recv):
#    rank = dist.get_rank()
#    size = dist.get_world_size()
#    send_buff = th.zeros(send.size())
#    recv_buff = th.zeros(send.size())
#    accum = th.zeros(send.size())
#    accum[:] = send[:]

#    left = ((rank - 1) + size) % size
#    right = (rank + 1) % size

#    for i in range(size - 1):
#        if i % 2 == 0:
#            # Send send_buff
#            send_req = dist.isend(send_buff, right)
#            dist.recv(recv_buff, left)
#            accum[:] += recv[:]
#        else:
#            # Send recv_buff
#            send_req = dist.isend(recv_buff, right)
#            dist.recv(send_buff, left)
#            accum[:] += send[:]
#        send_req.wait()
#    recv[:] = accum[:]


def allreduce(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = send.clone()
    recv_buff = send.clone()
    accum = send.clone()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv_buff[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send_buff[:]
        send_req.wait()
    recv[:] = accum[:]


def run_allreduce(rank, size):
    data = torch.ones(3) * (rank + 1)
    print(f'rank:{rank} data:{data}')
    recv = torch.zeros_like(data)
    allreduce(send=data, recv=recv)
    print(recv)
    """
        rank:2 data:tensor([3., 3., 3.])
        rank:0 data:tensor([1., 1., 1.])
        rank:3 data:tensor([4., 4., 4.])
        rank:1 data:tensor([2., 2., 2.])
        tensor([10., 10., 10.])
        tensor([10., 10., 10.])
        tensor([10., 10., 10.])
        tensor([10., 10., 10.])
        """


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_allreduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
