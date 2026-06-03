from configparser import ConfigParser
from copy import deepcopy
import csv
import datetime
import os
from poseestimate_mediapipe.module.movement import Movement
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import signal


@dataclass
class JointSeries:
    x: list[float]
    y: list[float]
    z: list[float]
    length: int
    jointname: str


class Corrector:

    def __init__(self, movement: Movement) -> None:
        self._movement = movement
        self._datatable = self._movement_to_datatable()
        self._colseriestable = self._get_colseriestable()
        self._jointseries_list = self._coltable_to_jointseries()

    def setconfig(self, config: ConfigParser):
        # LPF setting
        self.fps = config.getint('correct', 'fps')
        self.dt = 1.0 / self.fps
        self.fn = 1 / (2 * self.dt)
        self.fp = config.getfloat('correct', 'fp')
        self.fs = config.getfloat('correct', 'fs')
        self.gpass = config.getfloat('correct', 'gpass')
        self.gstop = config.getfloat('correct', 'gstop')

        # outlier setting
        self.threshold = config.getfloat('correct', 'threshold')
        self.median_window = config.getint('correct', 'median_window')

        # smoothing setting
        self.savgol_window = config.getint('correct', 'savgol_window')
        self.savgol_order = config.getint('correct', 'savgol_order')

        Wp = self.fp / self.fn
        Ws = self.fs / self.fn

        self.N, self.Wn = signal.buttord(Wp, Ws, self.gpass, self.gstop)
        self.b, self.a = signal.butter(self.N, self.Wn, "low")

    def _movement_to_datatable(self) -> list[list[float]]:
        # convert time series datarow(data origin of Pose3D or EmptyPose3D)
        return [datarow.get_flat() for datarow in self._movement.datastore]

    def _table_to_colseries(self, colnum: int) -> list[float]:
        return [datarow[colnum] for datarow in self._datatable]

    def _get_colseriestable(self) -> list[list[float]]:
        return [self._table_to_colseries(i)
                for i in range(len(self._movement.poseheader))]

    def _coltable_to_jointseries(self) -> list[JointSeries]:
        # 列データ用エイリアス
        cols_table = self._colseriestable

        # 列データ長さ
        length = len(cols_table[0])

        # 返却用リスト事前確保
        ret = []

        for index, name in enumerate(self._movement.landmarknames):
            jointnum = [index * 3 + 1 + axis for axis in range(3)]
            ret.append(JointSeries(cols_table[jointnum[0]],
                                   cols_table[jointnum[1]],
                                   cols_table[jointnum[2]],
                                   length, name))
        return ret

    def _check_invalid(self, dataseries: list[float], invalidset: set[float] = set([0.0, -1.0])) -> list[bool]:
        # 値が0ないし-1のものを補完する.
        # default invalidset is [-1.0, 0.0]
        return [False if val in invalidset else True for val in dataseries]

    def _check_outlier(self,
                       jointseries: JointSeries,
                       threshold: float,
                       median_window: int):

        jointseries_eachlist = [jointseries.x, jointseries.y, jointseries.z]
        move_med_xyz = [self._move_med(jointseries, median_window)
                        for jointseries in jointseries_eachlist]

        diffs = [self.sublist(series, move_median) for (
            series, move_median) in zip(jointseries_eachlist, move_med_xyz)]

        diffs_abs = [list(map(lambda val: val if val >= 0 else -val, li))
                     for li in diffs]

        judge = [self._median_diff_judge(
            diff_series, threshold) for diff_series in diffs_abs]

        return [(x and z)
                for (x, z) in zip(judge[0], judge[2])]

    def _median_diff_judge(self,
                           diffabs,
                           threshold: float = 0.15) -> list[bool]:
        return [False if value >= threshold else True for value in diffabs]

    def sublist(self, list_a: list[float], list_b: list[float]) -> list[float]:
        return [a - b for (a, b) in zip(list_a, list_b)]

    def interpolate_flags(self,
                          list_invalid: list[bool],
                          list_outlier: list[bool]) -> list[bool]:
        return [invalid and outlier
                for (invalid, outlier) in zip(list_invalid, list_outlier)]

    def _move_med(self, datalist: list[float], window: int) -> list[float]:
        vec = np.array(datalist)
        ret_ndarray = np.zeros_like(datalist)
        idx = np.arange(len(datalist))
        for i in range(len(datalist)):
            kernel = np.where(
                (idx >= (idx[i] - window)) & (idx <= (idx[i] + window)),
                True, False)
            ret_ndarray[i] = np.median(vec[kernel])
        return ret_ndarray.tolist()

    def run(self) -> None:
        self.new_colslisttable = deepcopy(self._colseriestable)

        for index, jointseries in enumerate(self._jointseries_list):
            print(f'now correct : {jointseries.jointname}')

            # invalid check
            invalid_list = self._check_invalid(jointseries.x, set([0.0, -1.0]))

            # outlier check
            outlier_list = self._check_outlier(
                jointseries, self.threshold, self.median_window)

            # interpolate flag check
            interpolate_list = self.interpolate_flags(
                invalid_list, outlier_list)

            nannumber = len(
                list(filter(lambda val: not val, interpolate_list)))
            invalidnumber = len(
                list(filter(lambda val: not val, invalid_list)))
            outliernumber = len(
                list(filter(lambda val: not val, outlier_list)))
            print(
                f'{jointseries.jointname}, interpolate NaN : {nannumber}, length : {jointseries.length}')
            print(
                f'invalid number : {invalidnumber}, outlier number : {outliernumber}')

            # pandas interpolate process
            pd_x_series = pd.Series(
                np.where(interpolate_list, jointseries.x, np.nan))
            pd_y_series = pd.Series(
                np.where(interpolate_list, jointseries.y, np.nan))
            pd_z_series = pd.Series(
                np.where(interpolate_list, jointseries.z, np.nan))

            # interpolate
            pd_x_series.interpolate(limit_direction='both',
                                    limit_area=None,
                                    limit=None,
                                    inplace=True,
                                    method='linear')
            pd_y_series.interpolate(limit_direction='both',
                                    limit_area=None,
                                    limit=None,
                                    inplace=True,
                                    method='linear')
            pd_z_series.interpolate(limit_direction='both',
                                    limit_area=None,
                                    limit=None,
                                    inplace=True,
                                    method='linear')

            # LPF
            # np_lpf_x = signal.filtfilt(self.b, self.a, pd_x_series.values)
            # np_lpf_y = signal.filtfilt(self.b, self.a, pd_y_series.values)
            # np_lpf_z = signal.filtfilt(self.b, self.a, pd_z_series.values)

            # FIR smoothing
            # future implement

            # smoothing (savitzky-Golay filter)
            np_lpf_x = signal.savgol_filter(pd_x_series.values, self.savgol_window, self.savgol_order)
            np_lpf_y = signal.savgol_filter(pd_y_series.values, self.savgol_window, self.savgol_order)
            np_lpf_z = signal.savgol_filter(pd_z_series.values, self.savgol_window, self.savgol_order)


            # save store
            self.new_colslisttable[3 * index + 1] = np_lpf_x.tolist()
            self.new_colslisttable[3 * index + 2] = np_lpf_y.tolist()
            self.new_colslisttable[3 * index + 3] = np_lpf_z.tolist()

    def csvwrite(self, outdir: str, filename: str, FPS, subject) -> None:
        framenum = len(self.new_colslisttable[0])
        dtnow = datetime.datetime.now()
        inforow = [filename,
                   'create_time', dtnow.strftime('%Y_%m_%d_%H_%M_%S'),
                   'FPS', FPS,
                   'Mean FPS', '',
                   'subject', subject,
                   'framenum', str(framenum)]

        with open(os.path.join(outdir, f'{filename}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(inforow)
            writer.writerow([])
            writer.writerow(self._movement.poseheader)

            # TODO これやるんだったらPose3Dオブジェクト作れる
            for row in range(len(self.new_colslisttable[0])):
                writtenrow = [self.new_colslisttable[col][row]
                              for col in range(len(self._movement.poseheader))]
                writer.writerow(writtenrow)

    def getmovement(self) -> Movement:
        retmovement = deepcopy(self._movement)
        datanumber = self._movement.landmarknames * 3

        # retmovement object list[pose3d] update process
        for row in range(len(retmovement.datastore)):
            posedata = [self.new_colslisttable[col + 1][row]
                        for col in range(len(datanumber))]
            retmovement.datastore[row].update(
                posedata, len(self._movement.landmarknames))

        return retmovement
