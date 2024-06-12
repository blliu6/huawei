import heapq
import sys

input = lambda: sys.stdin.readline().strip()


def getScores(cores, c):
    scores = 0
    for coreId, core_tasks in enumerate(cores):
        # line = str(len(core_tasks))
        nowTime = 0
        for i in range(1, len(core_tasks)):
            if core_tasks[i].msgType == core_tasks[i - 1].msgType:
                scores += 1
        for task in core_tasks:
            nowTime += task.exeTime
            if nowTime <= task.deadLine:
                scores += 1
    print('score is', scores)


class MessageTask:
    def __init__(self):
        self.msgType = 0
        self.usrInst = 0
        self.exeTime = 0
        self.deadLine = 0
        self.pos = 0


MAX_USER_ID = 10005


def main():
    # 1. 读取任务数、核数、系统最大执行时间
    n, m, c = map(int, input().split())

    # 2. 读取每个任务的信息
    tasks = [[] for _ in range(MAX_USER_ID)]
    task_total = []
    for _ in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        deadLine = min(deadLine, c)
        task = MessageTask()
        task.msgType, task.usrInst, task.exeTime, task.deadLine = msgType, usrInst, exeTime, deadLine
        task.pos = len(tasks[usrInst])
        tasks[usrInst].append(task)
        task_total.append(task)

    task_total.sort(key=lambda x: x.deadLine - x.exeTime)

    cores = [[] for _ in range(m)]
    usr_vis = [-1] * MAX_USER_ID
    cur_task = set()
    cur = [0 for _ in range(MAX_USER_ID)]  # 用户任务指针
    core_time = [0] * m

    # min_heap = [(0, i) for i in range(m)]
    # heapq.heapify(min_heap)

    for item in task_total:
        if item in cur_task:
            continue
        usr_id = item.usrInst

        if usr_vis[usr_id] != -1:
            core_idx = usr_vis[usr_id]
        else:
            # _, core_idx = heapq.heappop(min_heap)
            core_idx = 0
            for i in range(m):
                if core_time[i] < core_time[core_idx]:
                    core_idx = i
            usr_vis[usr_id] = core_idx

        for task in tasks[usr_id][cur[usr_id]:item.pos + 1]:
            cores[core_idx].append(task)
            cur_task.add(task)
            core_time[core_idx] += task.exeTime

        cur[usr_id] = item.pos + 1
        # heapq.heappush(min_heap, (core_time[core_idx], core_idx))

    for cores_tasks in cores:
        core_len = len(cores_tasks)
        cur_time = 0
        # 正向做一次
        for i in range(core_len):
            cur_time_old = cur_time
            cur_time += cores_tasks[i].exeTime
            if (i < core_len - 1 and cores_tasks[i].msgType == cores_tasks[i + 1].msgType) or (
                    i > 0 and cores_tasks[i].msgType == cores_tasks[i - 1].msgType):
                continue

            for j in range(i + 1, core_len):
                if cores_tasks[i].usrInst == cores_tasks[j].usrInst:
                    break

                if cur_time > cores_tasks[i].deadLine:
                    break

                if cores_tasks[i].msgType == cores_tasks[j].msgType:
                    temp = cores_tasks[i]
                    for k in range(i, j - 1):
                        cores_tasks[k] = cores_tasks[k + 1]
                    cores_tasks[j - 1] = temp
                    # i -= 1
                    break

                cur_time += cores_tasks[j].exeTime

            cur_time = cur_time_old + cores_tasks[i].exeTime

        # 反向做一次
        for i in range(core_len - 1, -1, -1):
            if (i < core_len - 1 and cores_tasks[i].msgType == cores_tasks[i + 1].msgType) or (
                    i > 0 and cores_tasks[i].msgType == cores_tasks[i - 1].msgType):
                continue

    # 4. 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        line = str(len(core_tasks))
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line + "\n")

    print(''.join(output_lines), end='')


if __name__ == "__main__":
    main()
