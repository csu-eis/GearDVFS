#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <iostream>
#include <vector>
#include <sstream>
typedef struct
{
    char *name;
    char *abbrev;
    int val;
} PerfEvent;

struct read_format
{
    uint64_t nr;
    struct
    {
        uint64_t value;
        uint64_t id;
    } values[];
};

const PerfEvent EVENT_LIST[] = {
    {"PERF_COUNT_HW_CPU_CYCLES", "cycles", PERF_COUNT_HW_CPU_CYCLES},
    {"PERF_COUNT_HW_INSTRUCTIONS", "instructions", PERF_COUNT_HW_INSTRUCTIONS},
    {"PERF_COUNT_HW_CACHE_REFERENCES", "cache-ref", PERF_COUNT_HW_CACHE_REFERENCES},
    {"PERF_COUNT_HW_CACHE_MISSES", "cache-miss", PERF_COUNT_HW_CACHE_MISSES},
    {"PERF_COUNT_HW_STALLED_CYCLES_FRONTEND", "stalled-cycles-front", PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
    {"PERF_COUNT_HW_STALLED_CYCLES_BACKEND", "stalled-cycles-back", PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
    {"PERF_COUNT_HW_BRANCH_MISSES", "branch-miss", PERF_COUNT_HW_BRANCH_MISSES},
};

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;
    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                  group_fd, flags);
    return ret;
}

void sys_perf(std::vector<int> cpus, std::vector<int> events, const int micro_seconds)
{

    size_t n_event = events.size();
    size_t n_cpu = cpus.size();
    int N_COUNTER = n_cpu * n_event;
    std::vector<std::vector<uint64_t>> result_list = std::vector<std::vector<uint64_t>>(n_cpu);

    // initialization
    struct perf_event_attr pea;
    struct read_format **rfs = (struct read_format **)calloc(n_cpu, sizeof(struct read_format *));
    int **fds = (int **)calloc(n_cpu, sizeof(int *));
    uint64_t **ids = (uint64_t **)calloc(n_cpu, sizeof(uint64_t *));
    char **bufs = (char **)calloc(n_cpu, sizeof(char *));

    for (int i = 0; i < n_cpu; i++)
    {
        fds[i] = (int *)calloc(N_COUNTER, sizeof(int));
        ids[i] = (uint64_t *)calloc(N_COUNTER, sizeof(uint64_t));
        bufs[i] = (char *)calloc(4096, sizeof(char));
        rfs[i] = (struct read_format *)bufs[i];
    }

    // for each cpu and each hardware event
    for (int i = 0; i < n_cpu; i++)
    {
        int cpu_index = cpus[i];
        for (int j = 0; j < n_event; j++)
        {
            int event_index = events[i];
            memset(&pea, 0, sizeof(struct perf_event_attr));
            pea.type = PERF_TYPE_HARDWARE;
            pea.size = sizeof(struct perf_event_attr);
            pea.config = EVENT_LIST[event_index].val;
            pea.disabled = 1;
            pea.exclude_kernel = 0;
            pea.exclude_hv = 1;
            pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
            if (j == 0)
            {
                fds[i][j] = syscall(__NR_perf_event_open, &pea, -1, cpu_index, -1, 0);
            }
            else
            {
                fds[i][j] = syscall(__NR_perf_event_open, &pea, -1, cpu_index, fds[i][0], 0);
            }
            if (fds[i][j] == -1)
            {
                fprintf(stderr, "Error opening leader %llx\n", pea.config);
                exit(EXIT_FAILURE);
            }
            ioctl(fds[i][j], PERF_EVENT_IOC_ID, &ids[i][j]);
        }
    }

    // monitoring for each cpu group
    for (int i = 0; i < n_cpu; i++)
    {
        ioctl(fds[i][0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds[i][0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    usleep(micro_seconds);
    for (int i = 0; i < n_cpu; i++)
    {
        ioctl(fds[i][0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }

    // read counters and pack into PyList
    for (int i = 0; i < n_cpu; i++)
    {
        read(fds[i][0], bufs[i], 4096 * sizeof(char));
        std::vector<uint64_t> cpu_result = std::vector<uint64_t>(n_event);
        for (int j = 0; j < n_event; j++)
        {
            // search for ids
            for (int k = 0; k < rfs[i]->nr; k++)
            {
                if (rfs[i]->values[k].id == ids[i][j])
                {
                    uint64_t val = rfs[i]->values[k].value;
                    cpu_result[j] = val;
                    // PyList_SetItem(cpu_result, j, Py_BuildValue("l", val));
                }
            }
            printf("cpu:%d, event: %s, count: %llu\n", cpus[i], EVENT_LIST[j].name, cpu_result[j]);
        }
        result_list[i] = cpu_result;
        // PyList_SetItem(result_list,cpu_result i, cpu_result);
    }

    // free spaces
    for (int i = 0; i < n_cpu; i++)
    {
        for (int j = 0; j < n_event; j++)
        {
            close(fds[i][j]);
        }
        free(fds[i]);
        free(ids[i]);
        free(bufs[i]);
    }
    free(fds);
    free(ids);
    free(bufs);
    free(rfs);
    // return result_list;
}

std::vector<int> split_num(std::string input)
{
    std::vector<int> tokens;
    std::string token;
    std::stringstream tokenStream(input);

    while (std::getline(tokenStream, token, ','))
    {
        tokens.push_back(std::stoi(token));
    }
    return tokens;
}

int main(int argc, char **argv)
{
    printf("hello sys_perf\n");

    if (argc < 4)
    {
        fprintf(stderr, "three args: cpus,evnets,duration(us)");
    }
    std::vector<int> cpus = split_num(argv[1]);
    std::vector<int> events = split_num(argv[2]);
    int duration = std::stoi(argv[3]);
    sys_perf(cpus, events, 100);
    return 0;
}