import heapq
from collections import defaultdict
import sys

input = lambda: sys.stdin.readline().strip()


def getScores(cores):
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


def opt(ops, pos, core_len, cores_tasks, sum_time):
    if ops == 0:  # 正向优化
        cur_time = 0
        # 正向做一次
        i = 0
        while i < core_len:
            cur_time_old = cur_time
            cur_time += cores_tasks[i].exeTime
            if (i < core_len - 1 and cores_tasks[i].msgType == cores_tasks[i + 1].msgType) or (
                    i > 0 and cores_tasks[i].msgType == cores_tasks[i - 1].msgType):
                i += 1
                continue

            vis = False
            out_time = (cur_time > cores_tasks[i].deadLine)
            # 移到j位置
            for j in range(i + 1, core_len - 1):
                if cores_tasks[i].usrInst == cores_tasks[j].usrInst:
                    break

                cur_time += cores_tasks[j].exeTime
                if cur_time > cores_tasks[i].deadLine and not out_time:
                    break

                if cores_tasks[i].msgType == cores_tasks[j + 1].msgType:
                    temp = cores_tasks[i]
                    for k in range(i, j):
                        cores_tasks[k] = cores_tasks[k + 1]
                    cores_tasks[j] = temp
                    for k in range(i, j + 1):
                        sum_time[pos][k] = sum_time[pos][k - 1] + cores_tasks[k].exeTime
                    vis = True
                    break

            if vis:
                cur_time = cur_time_old
            else:
                cur_time = cur_time_old + cores_tasks[i].exeTime
                i += 1
    elif ops == 1:  # 反向优化
        i = core_len - 1
        while i >= 0:
            if (i < core_len - 1 and cores_tasks[i].msgType == cores_tasks[i + 1].msgType) or (
                    i > 0 and cores_tasks[i].msgType == cores_tasks[i - 1].msgType):
                i -= 1
                continue

            for j in range(i - 2, -1, -1):
                if cores_tasks[i].usrInst == cores_tasks[j + 1].usrInst:
                    break

                # 任务j+1是否超时
                out_time = sum_time[pos][j + 1] > cores_tasks[j + 1].deadLine
                if sum_time[pos][j + 1] + cores_tasks[i].exeTime > cores_tasks[j + 1].deadLine and not out_time:
                    break

                if cores_tasks[i].msgType == cores_tasks[j].msgType:
                    temp = cores_tasks[i]
                    for k in range(i, j + 1, -1):
                        cores_tasks[k] = cores_tasks[k - 1]
                        sum_time[pos][k] = sum_time[pos][k - 1] + temp.exeTime
                    cores_tasks[j + 1] = temp
                    sum_time[pos][j + 1] = sum_time[pos][j] + cores_tasks[j + 1].exeTime
                    i += 1
                    break
            i -= 1
    elif ops == 2:  # 正向块优化
        i = 0
        while i < core_len - 1:
            if cores_tasks[i].msgType == cores_tasks[i + 1].msgType:
                j = i + 1
                usr = {cores_tasks[i].usrInst, cores_tasks[j].usrInst}
                ddl = [cores_tasks[i].deadLine - sum_time[pos][i], cores_tasks[j].deadLine - sum_time[pos][j]]
                while j + 1 < core_len and cores_tasks[i].msgType == cores_tasks[j + 1].msgType:
                    j += 1
                    usr.add(cores_tasks[j].usrInst)
                    ddl.append(cores_tasks[j].deadLine - sum_time[pos][j])

                ddl = [e for e in ddl if e >= 0]
                out_time = True if len(ddl) == 0 else False
                min_ddl = 0 if out_time else min(ddl)

                # 将j移到k位置
                total = 0
                vis = False
                for k in range(j + 1, core_len - 1):
                    if cores_tasks[k].usrInst in usr:
                        break
                    total += cores_tasks[k].exeTime
                    if total > min_ddl and not out_time:
                        break

                    if cores_tasks[i].msgType == cores_tasks[k + 1].msgType:
                        # 将[i,j]平移到[k-(j-i), k]
                        temp = cores_tasks[i:j + 1]
                        for z in range(i, k - (j - i)):
                            cores_tasks[z] = cores_tasks[z + j - i + 1]

                        for z in range(j - i + 1):
                            cores_tasks[k - (j - i) + z] = temp[z]

                        for z in range(i, k + 1):
                            sum_time[pos][z] = sum_time[pos][z - 1] + cores_tasks[z].exeTime
                        vis = True
                        break
                if not vis:
                    i += 1
            i += 1
    elif ops == 3:  # 反向块优化
        i = core_len - 1
        while i >= 0:
            if cores_tasks[i].msgType == cores_tasks[i - 1].msgType:
                j = i - 1
                usr = {cores_tasks[i].usrInst, cores_tasks[j].usrInst}
                total_exe = cores_tasks[i].exeTime + cores_tasks[j].exeTime
                while j - 1 >= 0 and cores_tasks[i].msgType == cores_tasks[j - 1].msgType:
                    j -= 1
                    usr.add(cores_tasks[j].usrInst)
                    total_exe += cores_tasks[j].exeTime

                # 将j移动到k
                vis = False
                for k in range(j - 1, 0, -1):
                    if cores_tasks[k].usrInst in usr:
                        break

                    if sum_time[pos][k] + total_exe > cores_tasks[k].deadLine:
                        break

                    # 将[j,i] 移动到 [k,k+(i-j)]
                    if cores_tasks[i].msgType == cores_tasks[k - 1].msgType:
                        temp = cores_tasks[j:i + 1]
                        for z in range(i, k + (i - j), -1):
                            cores_tasks[z] = cores_tasks[z - (i - j) - 1]

                        for z in range(i - j + 1):
                            cores_tasks[k + z] = temp[z]

                        for z in range(k, i + 1):
                            sum_time[pos][z] = sum_time[pos][z - 1] + cores_tasks[z].exeTime

                        vis = True
                        break
                if not vis:
                    i -= 1
            i -= 1


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

    k = min(3, m)
    core_dict = [defaultdict(int) for i in range(m)]
    for item in task_total:
        if item in cur_task:
            continue
        usr_id = item.usrInst

        if usr_vis[usr_id] != -1:
            core_idx = usr_vis[usr_id]
        else:
            # _, core_idx = heapq.heappop(min_heap)
            t = [(e, i) for i, e in enumerate(core_time)]
            t.sort(key=lambda x: x[0])

            task_type = item.msgType
            cnt = [(core_dict[t[i][1]][task_type], t[i][1]) for i in range(k)]  # 前k个核内任务类型和当前相同的数量和核的下标
            cnt.sort(key=lambda x: x[0], reverse=True)

            # core_idx = 0
            # for i in range(m):
            #     if core_time[i] < core_time[core_idx]:
            #         core_idx = i
            core_idx = cnt[0][1]
            usr_vis[usr_id] = core_idx

        for task in tasks[usr_id][cur[usr_id]:item.pos + 1]:
            cores[core_idx].append(task)
            cur_task.add(task)
            core_time[core_idx] += task.exeTime
            core_dict[core_idx][item.msgType] += 1

        cur[usr_id] = item.pos + 1
        # heapq.heappush(min_heap, (core_time[core_idx], core_idx))

    sum_time = [[] for _ in range(m)]
    for pos, cores_tasks in enumerate(cores):
        core_len = len(cores_tasks)

        sum_time[pos] = []
        total = 0
        for i in range(core_len):
            total += cores_tasks[i].exeTime
            sum_time[pos].append(total)

        opt(0, pos, core_len, cores_tasks, sum_time)
        opt(1, pos, core_len, cores_tasks, sum_time)
        opt(3, pos, core_len, cores_tasks, sum_time)
        opt(2, pos, core_len, cores_tasks, sum_time)
        opt(3, pos, core_len, cores_tasks, sum_time)
        opt(2, pos, core_len, cores_tasks, sum_time)
        opt(3, pos, core_len, cores_tasks, sum_time)
        # 处理超时任务
        # cur_time = 0
        # i = 0
        # while i < core_len:
        #     old_time = cur_time
        #     cur_time += cores_tasks[i].exeTime
        #     if cur_time <= cores_tasks[i].deadLine:
        #         i += 1
        #         continue
        #
        #     if (i < core_len - 1 and cores_tasks[i].msgType == cores_tasks[i + 1].msgType) or (
        #             i > 0 and cores_tasks[i].msgType == cores_tasks[i - 1].msgType):
        #         i += 1
        #         continue
        #
        #     for j in range(i + 1, core_len):
        #         # 处理必定超时任务，将i移到j-1
        #         if cores_tasks[j].usrInst == cores_tasks[i].usrInst:
        #             temp = cores_tasks[i]
        #             for k in range(i, j - 1):
        #                 cores_tasks[k] = cores_tasks[k + 1]
        #             cores_tasks[j - 1] = temp
        #             i -= 1
        #             break
        #     cur_time = old_time
        #     i += 1
        #
        # total = 0
        # for i in range(core_len):
        #     total += cores_tasks[i].exeTime
        #     sum_time[pos].append(total)
        #
        # i = core_len - 1
        # while i >= 0:
        #     if sum_time[pos][i] <= cores_tasks[i].deadLine:
        #         i -= 1
        #         continue
        #
        #     if (i < core_len - 1 and cores_tasks[i].msgType == cores_tasks[i + 1].msgType) or (
        #             i > 0 and cores_tasks[i].msgType == cores_tasks[i - 1].msgType):
        #         i -= 1
        #         continue
        #
        #     # 将i移动到j位置
        #     j = i - 1
        #     while j >= 0:
        #         if sum_time[pos][j] + cores_tasks[i].exeTime > cores_tasks[j].deadLine:
        #             break
        #
        #         if cores_tasks[j].usrInst == cores_tasks[i].usrInst:
        #             break
        #         j -= 1
        #     # 说明可以移动到j+1位置
        #     temp = cores_tasks[i]
        #     for k in range(i, j + 1, -1):
        #         cores_tasks[k] = cores_tasks[k - 1]
        #         sum_time[pos][k] = sum_time[pos][k - 1] + temp.exeTime
        #     cores_tasks[j + 1] = temp
        #     sum_time[pos][j + 1] = sum_time[pos][j] + cores_tasks[j + 1].exeTime
        #     if j + 1 < i:
        #         i -= 1

    # 4. 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        line = str(len(core_tasks))
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line + "\n")

    print(''.join(output_lines), end='')
    getScores(cores)


if __name__ == "__main__":
    main()
