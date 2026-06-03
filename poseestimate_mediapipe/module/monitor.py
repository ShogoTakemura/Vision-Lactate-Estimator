from pynput import keyboard


class Frame_monitor:
    def __init__(self, max_length, init_length=0):
        self._frame = init_length
        self._max_length = max_length
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._loopend_flag = False
        self._update_frame = True
        self._squat_start_frame = -1
        self._squat_end_frame = -1
        self._basemodel_frame_one = -1
        self._basemodel_frame_two = -1

    @property
    def update_frame(self):
        return self._update_frame

    @property
    def loopend_flag(self):
        return self._loopend_flag

    @property
    def frame(self):
        return self._frame

    @update_frame.setter
    def update_frame(self, value):
        self._update_frame = value

    @frame.setter
    def frame(self, value):
        try:
            input_value = int(value)
            self._frame = input_value
            self._clip_within_range()

        except ValueError as e:
            print(f'you input invalid value {e}')

    def _press_right(self):
        self._frame += 1
        self._clip_within_range()

    def _press_left(self):
        self._frame -= 1
        self._clip_within_range()

    def _press_up(self):
        self._frame -= 30
        self._clip_within_range()

    def _press_down(self):
        self._frame += 30
        self._clip_within_range()

    def _press_start(self):
        self._frame = 0

    def _press_end(self):
        self._frame = self._max_length - 1

    def _press_one(self):
        "初期フレーム保存用, 1ボタン押下時処理"
        self._squat_start_frame = self._frame

    def _press_two(self):
        "終了フレーム保存用, 2ボタン押下時処理"
        self._squat_end_frame = self._frame

    def _press_three(self):
        "基準フレーム保存用1, 3ボタン押下時処理"
        self._basemodel_frame_one = self._frame

    def _press_four(self):
        "基準フレーム保存用2, 4ボタン押下時処理"
        self._basemodel_frame_two = self._frame

    def _clip_within_range(self):
        if self._frame < 0:
            self._frame = 0
        elif self._frame > self._max_length - 1:
            self._frame = self._max_length - 1

    def _press_s(self, key):
        if (key.char == 's') or (key.char == 'S'):
            self._press_start()
            print(f'now frame is start frame : {self._frame}')
            self._update_frame = True

    def _press_e(self, key):
        if (key.char == 'e') or (key.char == 'E'):
            self._press_end()
            print(f'now frame is end frame : {self._frame}')
            self._update_frame = True

    def _press_arrow(self, key):
        if key == keyboard.Key.up:
            self._press_up()
            print(f'now frame is : {self._frame}')
            self._update_frame = True
        elif key == keyboard.Key.down:
            self._press_down()
            print(f'now frame is : {self._frame}')
            self._update_frame = True
        elif key == keyboard.Key.right:
            self._press_right()
            print(f'now frame is : {self._frame}')
            self._update_frame = True
        elif key == keyboard.Key.left:
            self._press_left()
            print(f'now frame is : {self._frame}')
            self._update_frame = True

    def _press_number(self, key):
        if key.char == '1':
            self._press_one()
            print(f'save squat init frame : {self._frame}')
            self._update_frame = True
        elif key.char == '2':
            self._press_two()
            print(f'save squat end frame : {self._frame}')
            self._update_frame = True
        elif key.char == '3':
            self._press_three()
            print(f'save basemodel frame 1 : {self._frame}')
            self._update_frame = True
        elif key.char == '4':
            self._press_four()
            print(f'save basemodel frame 2 : {self._frame}')
            self._update_frame = True

    def _on_press(self, key):
        try:
            print(f'Press key {key.char}')
            self._press_s(key)
            self._press_e(key)
            self._press_number(key)
        except AttributeError:
            print(f'Press key {key}')
            self._press_arrow(key)

    def _on_release(self, key):
        if key == keyboard.Key.esc:
            self._loopend_flag = True
            return False

    def _output_savedata(self) -> None:
        print('squat start frame : ', self._squat_start_frame)
        print('squat end frame : ', self._squat_end_frame)
        print('squat basemodel frame 1 : ', self._basemodel_frame_one)
        print('squat basemodel frame 2 : ', self._basemodel_frame_two)

    def _user_manual(self) -> None:
        print('Press key following below\n')
        print('Arrow down +30 frames, up -30 frames')
        print('Arrow right +1 frames, left -1 frames')
        print('Press 1~4 to save frame number')
        print('1: squat start frame, 2: squat end frame')
        print('3: squat basemodel 1 frame, 4: squat basemodel 2 frame')
        print('s : start, e : last, Esc : End stream\n')