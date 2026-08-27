# -*- coding: utf-8 -*-
class SetStepRedun:
    '''设置步长冗余'''
    '''设置步长'''
    @classmethod
    def set_step(cls, step, redun, step_spinbox, redun_spinbox):
        step_copy = step
        step = round(float(step_spinbox.value()))  # 获取步长
        if step <= redun:
            step = step_copy
        redun_spinbox.setRange(0, int(step) - 1)
        return step

    '''设置冗余'''
    @classmethod
    def set_redun(cls, step, redun, step_spinbox, redun_spinbox, size):
        redun_copy = redun
        redun = round(float(redun_spinbox.value()))  # 获取步长
        if step <= redun:
            redun = redun_copy
        step_spinbox.setRange(int(redun) + 1, int(size))
        return redun

    '''更新框架尺寸'''
    @classmethod
    def update_box_size(cls, step_list, size, origin, index):
        x1, y1, z1 = origin
        box_size = [x1, x1 + step_list[0], y1, y1 + step_list[1], z1, z1 + step_list[2]]
        if origin[index] + step_list[index] >= size:
            box_size[index * 2], box_size[index * 2 + 1] = size - step_list[index], size
            origin[index] = size - step_list[index]
            up_over = True
        else:
            up_over = False

        return box_size, origin, up_over

    '''获取层数'''
    @classmethod
    def get_layers(cls, step, redun, size, index):  # 获取层数
        # 获取轴层数
        num = 1
        d = step - redun
        while step < size:
            num += 1
            step += d
        if index == 0:
            print("x轴层数:", num)
        if index == 1:
            print("y轴层数:", num)
        if index == 2:
            print("z轴层数:", num)
        return num

    '''上切'''
    @classmethod
    def box_to_up(cls, step, redun, box_size, origin, size, up_over, index):  # 上切
        if not box_size[index * 2 + 1] == size:
            print("###")
            print("变化前：")
            print("Origin(x,y,z) =", origin)
            print("Size(x:,y:,z:) =", box_size)
            add = abs(step - redun)  # 增加距离
            box_size[index * 2] += add
            box_size[index * 2 + 1] += add
            if box_size[index * 2 + 1] >= size:  # 超出或贴近尺寸
                box_size[index * 2 + 1] = size
                box_size[index * 2] = size - step
                origin[index] = size - step
                up_over = True
            else:
                origin[index] += add
            print("变化后：")
            print("Origin(x,y,z) =", origin)
            print("Size(x:,y:,z:) =", box_size)
            print("###")
        down_over = False
        return box_size, origin, up_over, down_over

    '''下切'''
    @classmethod
    def box_to_down(cls, step, redun, box_size, origin, down_over, index):  # 下切
        if not box_size[index * 2] == 0:
            print("###")
            print("变化前：")
            print("Origin(x,y,z) =", origin)
            print("Size(x:,y:,z:) =", box_size)
            reduce = abs(step - redun)  # 增加距离
            box_size[index * 2] -= reduce
            box_size[index * 2 + 1] -= reduce
            if box_size[index * 2] <= 0:  # 超出或贴近尺寸
                box_size[index * 2 + 1] = step
                box_size[index * 2] = 0
                origin[index] = 0
                down_over = True
            else:
                origin[index] -= reduce
            print("变化后：")
            print("Origin(x,y,z) =", origin)
            print("Size(x:,y:,z:) =", box_size)
            print("###")
        up_over = False
        return box_size, origin, up_over, down_over
