import math
import mindspore as ms
import numpy as np
from mindspore import Tensor, ops, nn
from mindspore.common.parameter import Parameter
import platform

# 64 16 64

attention_probs_dropout_prob = 0.3
num_attention_heads = 16
INF = 1. * 1e9
initializer_range = 0.02
compute_type = ms.float16
use_one_hot_embeddings = False
batch_size = 96
num_hidden_layers = 1
embedding_size = 1024
hidden_size = 1024
intermediate_size = 4096
max_position_embeddings = 128
hidden_dropout_prob = 0.3
hidden_act = "relu"
vocab_size = 36560
plat = platform.system().lower()
if plat == 'windows':
    print("当前系统:windows10")
    device_target = "CPU"
elif plat == 'linux':
    print("当前系统:linux")
    device_target = "CPU"
else:
    raise SystemError("未受支持的系统!")


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return Tensor(norm)


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        depth (int): Hidden size.
        min_timescale (float): Default: 1.
        max_timescale (float): Default: 10000.

    Returns:
        Tensor of shape (length, depth)
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


def CreateMask(input_mask):
    input_shape = ops.Shape()(input_mask)
    shape_right = (input_shape[0], 1, input_shape[1])
    shape_left = input_shape + (1,)
    input_mask = ops.Cast()(input_mask, ms.float32)
    mask_left = ops.Reshape()(input_mask, shape_left)
    mask_right = ops.Reshape()(input_mask, shape_right)
    attention_mask = ops.BatchMatMul()(mask_left, mask_right)
    return attention_mask


def EmbedLookup(vocab_size, embedding_size, use_one_hot_embeddings, initializer_range, input_ids):
    input_shape = ops.Shape()(input_ids)
    flat_ids = ops.Reshape()(input_ids, (-1,))
    embedding_table = Parameter(normal_weight([vocab_size, embedding_size], embedding_size))
    if use_one_hot_embeddings:
        one_hot_ids = ops.OneHot()(flat_ids, vocab_size, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        output_for_reshape = ops.MatMul()(one_hot_ids, embedding_table)
    else:
        output_for_reshape = ops.Gather()(embedding_table, flat_ids, 0)

    out_shape = input_shape + (embedding_size,)
    output = ops.Reshape()(output_for_reshape, out_shape)
    return output, embedding_table.value()


class Embedding_Lookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """

    def __init__(self, vocab_size, embedding_size, use_one_hot_embeddings=False, initializer_range=0.02):
        super(Embedding_Lookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(normal_weight([vocab_size, embedding_size], embedding_size))
        self.expand = ops.ExpandDims()
        self.shape_flat = (-1,)
        self.gather = ops.Gather()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.array_mul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def construct(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        input_shape = self.shape(input_ids)

        flat_ids = self.reshape(input_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = self.reshape(output_for_reshape, out_shape)
        return output, self.embedding_table.value()


class EmbeddingPostprocessor(nn.Cell):
    def __init__(self,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=128,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = Tensor([math.sqrt(float(embedding_size))], dtype=ms.float32)
        self.multiply = ops.Mul()
        self.add = ops.Add()
        self.dropout = nn.Dropout(p=1 - dropout_prob, dtype=ms.float32)
        self.use_dropout = dropout_prob > 0
        self.expand_dims = ops.ExpandDims()
        self.position_embedding_table = Tensor(position_encoding(max_position_embeddings, embedding_size),
                                               ms.float32)
        self.shape = ops.Shape()

    def construct(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        input_shape = self.shape(word_embeddings)
        input_len = input_shape[1]

        output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(output, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output


class endecoder(nn.Cell):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings,
                 initializer_range=0.02):
        super(endecoder, self).__init__()
        self.embed_lookup = Embedding_Lookup(vocab_size=vocab_size, embedding_size=embedding_size,
                                             use_one_hot_embeddings=use_one_hot_embeddings,
                                             initializer_range=initializer_range)
        self.embedPostProcess = EmbeddingPostprocessor(embedding_size=embedding_size,
                                                       use_one_hot_embeddings=use_one_hot_embeddings,
                                                       initializer_range=0.02,
                                                       max_position_embeddings=max_position_embeddings,
                                                       dropout_prob=hidden_dropout_prob)
        self.encoderstack = EncoderStack(num_attention_heads=num_attention_heads,
                                         intermediate_size=intermediate_size,
                                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                                         use_one_hot_embeddings=use_one_hot_embeddings,
                                         initializer_range=initializer_range,
                                         hidden_dropout_prob=hidden_dropout_prob,
                                         hidden_act=hidden_act,
                                         compute_type=compute_type,
                                         num_hidden_layers=num_hidden_layers,
                                         batch_size=batch_size,
                                         hidden_size=hidden_size)
        self.decoderStack = DecoderStack(batch_size=batch_size,
                                         hidden_size=hidden_size,
                                         num_attention_heads=num_attention_heads,
                                         num_hidden_layers=num_hidden_layers,
                                         intermediate_size=intermediate_size,
                                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                                         use_one_hot_embeddings=use_one_hot_embeddings,
                                         initializer_range=initializer_range,
                                         hidden_dropout_prob=hidden_dropout_prob,
                                         hidden_act=hidden_act,
                                         compute_type=compute_type)

    def construct(self, source_ids, source_mask, target_ids, target_mask):
        seq_length = ops.Shape()(source_ids)[1]
        src_word_embeddings, embedding_tables = self.embed_lookup(source_ids)
        src_embedding_output = self.embedPostProcess(src_word_embeddings)
        enc_attention_mask = CreateMask(source_mask)
        encoder_output = self.encoderstack(CastWrapper(dst_type=compute_type)(src_embedding_output),
                                           CastWrapper(dst_type=compute_type)(enc_attention_mask),
                                           seq_length
                                           )
        future_mask = convert_np_to_tensor_encoder(seq_length)
        tgt_word_embeddings, _ = self.embed_lookup(input_ids=target_ids)
        """Postprocessors apply positional embeddings to word embeddings."""
        tgt_attention_mask = CreateMask(target_mask)
        tgt_attention_mask = ops.Mul()(tgt_attention_mask, ops.ExpandDims()(future_mask, 0))
        # transformer decoder
        decoder_output = self.decoderStack(CastWrapper(dst_type=compute_type)(tgt_word_embeddings),
                                           CastWrapper(dst_type=compute_type)(tgt_attention_mask),
                                           encoder_output, enc_attention_mask,
                                           seq_length, seq_length)

        return decoder_output


def _average_units(shape):
    """
    Average shape dim.
    """
    if not shape:
        return 1.
    if len(shape) == 1:
        return float(shape[0])
    if len(shape) == 2:
        return float(shape[0] + shape[1]) / 2.
    raise RuntimeError("not support shape.")


def weight_variable(shape):
    scale_shape = shape
    avg_units = _average_units(scale_shape)
    scale = 1.0 / max(1., avg_units)
    limit = math.sqrt(3.0 * scale)
    values = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(values)


class MultiheadAttention(nn.Cell):
    def __init__(self,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 out_tensor_width,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 out_act=None,
                 has_attention_mask=True,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=True,
                 compute_type=ms.float32):
        super(MultiheadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor

        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = ops.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    has_bias=True,
                                    weight_init=weight_variable([units, from_tensor_width])).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  has_bias=True,
                                  weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    has_bias=True,
                                    weight_init=weight_variable([units, to_tensor_width])).to_float(compute_type)
        self.out_layer = nn.Dense(units,
                                  out_tensor_width,
                                  activation=out_act,
                                  has_bias=True,
                                  weight_init=weight_variable([out_tensor_width, units])).to_float(compute_type)

        self.matmul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.multiply = ops.Mul()
        self.transpose = ops.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = Tensor([-10000.0, ], dtype=compute_type)
        self.batch_num = batch_size * num_attention_heads
        self.matmul = ops.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=1 - attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = ops.ExpandDims()
            self.sub = ops.Sub()
            self.add = ops.Add()
            self.cast = ops.Cast()
            self.get_dtype = ops.DType()

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.softmax_cast = ops.Cast()

    def construct(self, from_tensor, to_tensor, seq_length, enc_seq_length, attention_mask=None):
        """Apply multihead attention."""
        from_seq_length = seq_length
        to_seq_length = enc_seq_length
        shape_from = (self.batch_size, from_seq_length, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, to_seq_length, self.num_attention_heads, self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (self.batch_size * from_seq_length, self.num_attention_heads * self.size_per_head)
            if from_seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)

        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)
        # print("query_out", query_out.shape, shape_from)
        query_layer = self.reshape(query_out, shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(ops.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, ms.float32)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key_layer))
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, shape_return)
        context_layer = self.out_layer(context_layer)
        return context_layer


class LayerPreprocess(nn.Cell):
    """
    preprocess input of each layer.
    """

    def __init__(self,
                 in_channels=None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm((in_channels,))
        self.cast = ops.Cast()
        """
            Returns a tensor with the new specified data type.
            Examples:
                >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
                >>> input_x = Tensor(input_np)
                >>> type_dst = mindspore.int32
                >>> cast = ops.Cast()
                >>> output = cast(input_x, type_dst)
                >>> print(output.dtype)
                Int32
                >>> print(output.shape)
                (2, 3, 4, 5)
            """
        self.get_dtype = ops.DType()

    def construct(self, input_tensor):
        output = self.cast(input_tensor, ms.float32)
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output


class SelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        batch_size (int): Batch size of input dataset.
        from_seq_length (int): Length of query sequence.
        to_seq_length (int): Length of memory sequence.
        hidden_size (int): Size of attention layers.
        num_attention_heads (int): Number of attention heads. Default: 16.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        has_attention_mask (bool): Specifies whether has attention mask. Default: True.
        is_encdec_att (bool): Specifies whether query sequence and memory sequence are different. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: ms.float32.
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_attention_heads=16,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 has_attention_mask=True,
                 is_encdec_att=False,
                 compute_type=ms.float32):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.is_encdec_att = is_encdec_att

        self.attention = MultiheadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, memory_tensor, attention_mask, seq_length, enc_seq_length):
        """Apply self-attention."""
        input_tensor = self.reshape(input_tensor, self.shape)
        memory_tensor = self.reshape(memory_tensor, self.shape)

        output = self.preprocess(input_tensor)

        if not self.is_encdec_att:
            memory_tensor = output

        attention_output = self.attention(output, memory_tensor, seq_length, enc_seq_length, attention_mask)
        output = self.postprocess(attention_output, input_tensor)
        return output


class LayerPostprocess(nn.Cell):
    """
    postprocess output of each layer.
    """

    def __init__(self,
                 dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()
        self.add = ops.Add()
        self.dropout = nn.Dropout(p=1 - dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor, input_tensor):
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class EncoderCell(nn.Cell):
    """
    Encoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers. Default: 1024.
        seq_length (int): Length of input sequence. Default: 128.
        num_attention_heads (int): Number of attention heads. Default: 16.
        intermediate_size (int): Size of intermediate layer. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.1.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: ms.float32.
    """

    def __init__(self,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            is_encdec_att=False,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states, attention_mask, seq_length):
        # self-attention with ln, res
        attention_output = self.attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward

    Args:
        in_channels (int): Size of the input layer.
        hidden_size (int): Size of the hidden layer.
        out_channels (int): Size of the output layers.
        hidden_act (str): name of the activation function. Default: relu
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in FeedForward. Default: ms.float32.
    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 hidden_act="relu",
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 compute_type=ms.float32):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Dense(in_channels,
                              hidden_size,
                              activation=hidden_act,
                              weight_init=weight_variable([hidden_size, in_channels])).to_float(compute_type)
        self.conv2 = nn.Dense(hidden_size,
                              out_channels,
                              weight_init=weight_variable([out_channels, hidden_size])).to_float(compute_type)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = ops.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(p=1 - hidden_dropout_prob)
        self.use_dropout = hidden_dropout_prob > 0

    def construct(self, input_tensor):
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor)
        output = self.conv1(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return Tensor(np.tril(ones), dtype=ms.float32)


class EncoderStack(nn.Cell):
    """
    Multi-layer transformer encoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        seq_length (int): Length of input sequence.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1..
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: ms.float32.
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(EncoderStack, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        # self.num_hidden_layers = num_hidden_layers
        # self.batch_size = batch_size
        # self.hidden_size = hidden_size
        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.reshape = ops.Reshape()
        """
            Rearranges the input Tensor based on the given shape.
            Examples:
                >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
                >>> reshape = ops.Reshape()
                >>> output = reshape(input_x, (3, 2))
                >>> print(output)
                [[-0.1  0.3]
                 [ 3.6  0.4]
                 [ 0.5 -3.2]]
            """

        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask, seq_length):
        """Apply encoder."""
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output
        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """

    def __init__(self, src_type=ms.float32, dst_type=ms.float32):
        super(CastWrapper, self).__init__()
        self.cast = ops.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)


class Mod(nn.Cell):
    """
    Mod function.

    Args:
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: ms.float32.
    """

    def __init__(self,
                 compute_type=ms.float32):
        super(Mod, self).__init__()
        self.compute_type = compute_type
        self.floor_div = ops.FloorDiv()
        self.sub = ops.Sub()
        self.multiply = ops.Mul()

    def construct(self, input_x, input_y):
        x = self.floor_div(input_x, input_y)
        x = self.multiply(x, input_y)
        x = self.sub(input_x, x)
        return x


class LengthPenalty(nn.Cell):
    """
    Normalize scores of translations according to their length.

    Args:
        weight (float): Weight of length penalty. Default: 1.0.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: ms.float32.
    """

    def __init__(self,
                 weight=1.0,
                 compute_type=ms.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight
        self.add = ops.Add()
        self.pow = ops.Pow()
        self.div = ops.RealDiv()
        self.cast = ops.Cast()
        self.five = Tensor(5.0, ms.float32)
        self.six = Tensor(6.0, ms.float32)

    def construct(self, length_tensor):
        length_tensor = self.cast(length_tensor, ms.float32)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output


class DecoderCell(nn.Cell):
    """
    decoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the Transformer decoder layers. Default: 1024.
        seq_length (int): Length of input sequence. Default: 128.
        enc_seq_length (int): Length of source sentences. Default:128
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: ms.float32.
    """

    def __init__(self,
                 batch_size,
                 hidden_size=1024,
                 num_attention_heads=12,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=ms.float32):
        super(DecoderCell, self).__init__()
        self.self_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=False,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.cross_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=True,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,  # 1024
            hidden_size=intermediate_size,  # 4096
            out_channels=hidden_size,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        # self-attention with ln, res
        attention_output = self.self_attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # cross-attention with ln, res
        attention_output = self.cross_attention(attention_output, enc_states, enc_attention_mask,
                                                seq_length, enc_seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class DecoderStack(nn.Cell):
    def __init__(self, batch_size, hidden_size, num_hidden_layers, num_attention_heads=16, intermediate_size=4096,
                 attention_probs_dropout_prob=0.1, use_one_hot_embeddings=False, initializer_range=0.02,
                 hidden_dropout_prob=0.1, hidden_act="relu", compute_type=ms.float32):
        super(DecoderStack, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        layers = []
        for _ in range(num_hidden_layers):
            layer = DecoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.reshape = ops.Reshape()
        self.shape = (-1, hidden_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def construct(self, input_tensor, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        """Apply decoder."""
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, enc_states, enc_attention_mask,
                                        seq_length, enc_seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        print("prev_output", prev_output.shape)
        output = self.reshape(prev_output, out_shape)
        return output


if __name__ == '__main__':
    ms.set_context(device_target=device_target)
    print("running on", device_target)
    if device_target == "CPU":
        ms.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")
    a = np.random.randint(0, 8000, (96, 16))
    b = np.random.randint(0, 1, (96, 16))
    c = np.random.randint(0, 1, (96, 16))
    d = np.random.randint(0, 8600, (96, 16))
    a, b, c, d = Tensor(a, ms.int32), Tensor(b, ms.int32), Tensor(c, ms.int32), Tensor(d, ms.int32)
    print("shape")
    print(a.shape, b.shape, c.shape, d.shape)
    ed = endecoder(vocab_size, embedding_size, use_one_hot_embeddings, initializer_range)
    print(ed(a, b, c, d).shape)
