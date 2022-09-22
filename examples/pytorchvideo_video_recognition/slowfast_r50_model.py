from pytorchvideo.models.hub.slowfast import slowfast_r50


class SlowFast(type):
    def __new__(cls):

        """

        input_clip_length = 32
        input_crop_size = 224
        input_channel = 3

        model = create_slowfast(
            slowfast_channel_reduction_ratio=8,
            slowfast_conv_channel_fusion_ratio=2,
            slowfast_fusion_conv_kernel_size=(7, 1, 1),
            slowfast_fusion_conv_stride=(4, 1, 1),
            input_channels=(input_channel,) * 2,
            model_depth=18,
            model_num_class=400,
            dropout_rate=0,
        )
        #model = create_resnet()
        """
        model = slowfast_r50()

        return model
