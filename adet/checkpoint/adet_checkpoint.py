import os
import pickle
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import torch
from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.file_io import PathManager


class _IncompatibleKeys(
    NamedTuple(
        # pyre-fixme[10]: Name `IncompatibleKeys` is used but not defined.
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            ("incorrect_shapes", List[Tuple]),
        ],
    )
):
    pass


class AdetCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectronCheckpointer`, but is able to convert models
    in AdelaiDet, such as LPF backbone.
    """

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if incompatible is not None:  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore

        # return any further checkpoint data
        return checkpoint

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:  # pyre-ignore
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    print(shape_checkpoint)
                    add_shape = (1,)
                    for i in range(len(shape_checkpoint)):
                        if i == 0:
                            continue
                        add_shape += (shape_checkpoint[i],)
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][70], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][83], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][74], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][82], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][88], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][87], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][65], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][87], add_shape))
                    )
                    checkpoint_state_dict[k] = torch.cat(
                        (checkpoint_state_dict[k], torch.reshape(checkpoint_state_dict[k][68], add_shape))
                    )
                    # for i in range(9):
                    #    checkpoint_state_dict[k] = torch.cat((checkpoint_state_dict[k],(0.1**0.5)*torch.randn(add_shape)))
                    # print(checkpoint_state_dict[k].shape)
                    # incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    # checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)
