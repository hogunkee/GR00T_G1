# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
from pathlib import Path

import av
import gymnasium as gym
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: list[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int | None = 0,
    allow_negative: bool = False,
) -> tuple[list[int], list[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = np.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


class MultiVideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self.cameras = ['bird', 'front', 'back', 'side']
        self._reset_state()

    def _reset_state(self):
        self.container = [None for c in self.cameras]
        self.stream = [None for c in self.cameras]
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile:v": "high"},
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream[0] is not None
        #return self.stream is not None

    def start(self, file_paths, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        for cam_idx in range(len(self.cameras)):
            self.container[cam_idx] = av.open(file_paths[cam_idx], mode="w")
            self.stream[cam_idx] = self.container[cam_idx].add_stream(self.codec, rate=self.fps)
            codec_context = self.stream[cam_idx].codec_context
            for k, v in self.kwargs.items():
                setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, imgs: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        for cam_idx in range(len(self.cameras)):
            if self.shape is None:
                self.shape = imgs[cam_idx].shape
                self.dtype = imgs[cam_idx].dtype
                h, w, c = imgs[cam_idx].shape
                self.stream[cam_idx].width = w
                self.stream[cam_idx].height = h
            assert imgs[cam_idx].shape == self.shape
            assert imgs[cam_idx].dtype == self.dtype

            frame = av.VideoFrame.from_ndarray(imgs[cam_idx], format=self.input_pix_fmt)
            for i in range(n_repeats):
                for packet in self.stream[cam_idx].encode(frame):
                    self.container[cam_idx].mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        for cam_idx in range(len(self.cameras)):
            # Flush stream
            for packet in self.stream[cam_idx].encode():
                self.container[cam_idx].mux(packet)

            # Close the file
            self.container[cam_idx].close()

        # reset runtime parameters
        self._reset_state()


class MultiVideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recorder: MultiVideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        cameras=['birdview', 'frontview', 'backview', 'sideview'],
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.cameras = cameras
        self.file_paths = None

        self.step_count = 0

        self.is_success = False

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recorder.stop()

        if self.video_dir is not None and self.file_paths is not None:
            # rename the file to indicate success or failure
            for cidx in range(len(self.cameras)):
                original_filestem = self.file_paths[cidx].stem
                new_filestem = f"{original_filestem}_success{int(self.is_success)}"
                new_file_path = self.video_dir / f"{new_filestem}.mp4"
                os.rename(self.file_paths[cidx], new_file_path)

        self.is_success = False
        if self.video_dir is not None:
            self.file_paths = [self.video_dir / f"{uuid.uuid4()}_{cam}.mp4" for cam in self.cameras]
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_paths is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_paths)

            #frame = self.env.render()
            frames = []
            for cam in self.cameras: 
                frame = self.env.env.env.env.sim.render(camera_name=cam, width=640, height=480, depth=False)[::-1]
                frames.append(frame)
                assert frame.dtype == np.uint8
            self.video_recorder.write_frame(frames)
            self.is_success = result[-1]["success"]
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_paths
