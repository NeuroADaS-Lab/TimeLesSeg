import torch
import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager
from time import sleep

from timelesseg.preprocessing.preprocessing import preprocess_case


def preprocess_fromfiles_save_to_queue(
    list_of_case_dicts: list[dict[str, tuple[str | None] | str | None]],
    outfiles: list[str] | None,
    preprocessing_kwargs: dict,
    target_queue: Queue,
    done_event: Event,
    abort_event: Event
):
    try:
        for i, case_dict in enumerate(list_of_case_dicts):
            data, seg, data_properties = preprocess_case(image_paths=case_dict['images'],
                                                         seg_path=None, # removed seg. Doesn't make any sense
                                                         preprocessing_kwargs=preprocessing_kwargs)
            del seg

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {
                'data': data,
                'data_properties': data_properties,
                'ofile': outfiles[i] if outfiles is not None else None,
                'identifier': case_dict.get('identifier', case_dict['images'][0]) # fall back to image path if not identifier is provided
            }

            # try to put results of preprocessing call to queue. If it is full loop until it isn't
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    # try to put item in queue. If full, wait of 0.01 seconds to try again.
                    # if it still is full after timeout seconds, put will raise a Full Exception
                    # which is caught afterwards
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass

        # once everything has been processed, signal to main loop (see below) that the current worker is done.
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(
    list_of_case_dicts: list[dict[str, tuple[str | None] | str | None]],
    outfiles: list[str] | None,
    preprocessing_kwargs: dict,
    num_processes: int,
    pin_memory: bool = False
):

    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_case_dicts), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []

    # panic button. If one worker crashes, we set this so everyone else stops.
    # we only need one since we don't care which worker crashes, only that there has been an error
    abort_event = manager.Event()

    # We start N processes, each with its dedicated queue and "done event"
    for i in range(num_processes):
        event = manager.Event()
        # maxsize=1 forces the worker to pause if it has produced 
        # 1 item that hasn't been picked up yet. This prevents RAM explosion.
        queue = Manager().Queue(maxsize=1)

        # Distribute the work using Python Slicing (Round Robin). If num_processes=2:
        # Worker 0 gets index [0, 2, 4, 6...]
        # Worker 1 gets index [1, 3, 5, 7...]
        args=(
            list_of_case_dicts[i::num_processes],
            outfiles[i::num_processes] if outfiles is not None else None,
            preprocessing_kwargs,
            queue,
            event,
            abort_event
        )
        # Create and fire the background process
        pr = context.Process(target=preprocess_fromfiles_save_to_queue, args=args, daemon=True)
        pr.start()

        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    # worker_ctr is not incremented as w+=1. It goes from 0 to num_processes - 1
    # it essentially helps us check whether each worker is done or its queue is empty
    # and helps us return the correct order of preprocessed results/items (Round Robin slicing)
    worker_ctr = 0
    # loop for worker w until it is done (first check) and its queue is empty:
    # not A or not B == not (A and B) 
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # if target queue of current worker is not empty, empty it – i.e., get its item to yield it afterwards
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes # move on to next worker
        else: # target queue is empty. We cannot return an item since there is nothing to return (the queue is empty), hence the continue later
            # all workers are okay if: either they are alive or they are done, and the abort event has not been set
            all_ok = (
                all([i.is_alive() or j.is_set() for i, j in zip(processes, done_events)])
                and not abort_event.is_set()
            )
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item

    # once everything has been processed, close all processes cleanly.
    [p.join() for p in processes]
