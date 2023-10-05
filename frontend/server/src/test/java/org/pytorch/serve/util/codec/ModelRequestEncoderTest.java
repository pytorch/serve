package org.pytorch.serve.util.codec;

import static org.testng.Assert.assertEquals;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandler;
import io.netty.channel.embedded.EmbeddedChannel;
import java.io.IOException;
import java.util.ArrayList;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.RequestInput;
import org.testng.annotations.Test;

public class ModelRequestEncoderTest {

    @Test
    public void testEmpty() {
        ChannelHandler encoder = new ModelRequestEncoder(false);
        EmbeddedChannel channel = new EmbeddedChannel(encoder);
        ModelInferenceRequest msg = new ModelInferenceRequest("testModel");
        writeToChannelAndFlush(channel, msg);
        byte[] expected =
                new byte[] {
                    'I', (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF,
                };
        assertOutboundEquals(channel, expected);
    }

    @Test
    public void testSimple() throws IOException {
        ChannelHandler encoder = new ModelRequestEncoder(true);
        EmbeddedChannel channel = new EmbeddedChannel(encoder);
        ModelInferenceRequest msg = new ModelInferenceRequest("testModel");
        ArrayList list = new ArrayList<RequestInput>();
        RequestInput input = new RequestInput("request_id");
        byte[] compressed =
                new byte[] {
                    '{', '\"', 'd', 'a', 't', 'a', '\"', ':', ' ', '\"', 'v', 'a', 'l', 'u', 'e',
                    '\"', '}',
                };

        input.addParameter(new InputParameter("input_name", compressed, "application/json"));
        list.add(input);
        msg.setRequestBatch(list);
        writeToChannelAndFlush(channel, msg);

        byte[] expected =
                new byte[] {
                    'I',
                    0x00,
                    0x00,
                    0x00,
                    0x0A,
                    'r',
                    'e',
                    'q',
                    'u',
                    'e',
                    's',
                    't',
                    '_',
                    'i',
                    'd',
                    (byte) 0xFF,
                    (byte) 0xFF,
                    (byte) 0xFF,
                    (byte) 0xFF, // end of headers
                    0x00,
                    0x00,
                    0x00,
                    0x0A,
                    'i',
                    'n',
                    'p',
                    'u',
                    't',
                    '_',
                    'n',
                    'a',
                    'm',
                    'e',
                    0x00,
                    0x00,
                    0x00,
                    0x10,
                    'a',
                    'p',
                    'p',
                    'l',
                    'i',
                    'c',
                    'a',
                    't',
                    'i',
                    'o',
                    'n',
                    '/',
                    'j',
                    's',
                    'o',
                    'n',
                    0x00,
                    0x00,
                    0x00,
                    (byte) 0x11,
                    '{',
                    '\"',
                    'd',
                    'a',
                    't',
                    'a',
                    '\"',
                    ':',
                    ' ',
                    '\"',
                    'v',
                    'a',
                    'l',
                    'u',
                    'e',
                    '\"',
                    '}',
                    (byte) 0xFF,
                    (byte) 0xFF,
                    (byte) 0xFF,
                    (byte) 0xFF, // end of parameters
                    (byte) 0xFF,
                    (byte) 0xFF,
                    (byte) 0xFF,
                    (byte) 0xFF // end of batch
                };
        assertOutboundEquals(channel, expected);
    }

    private void assertOutboundEquals(EmbeddedChannel channel, byte[] expected) {
        ByteBuf buf = channel.readOutbound();
        byte[] actual = new byte[expected.length];
        try {
            buf.readBytes(actual, 0, expected.length);
        } finally {
            buf.release();
        }
        assertEquals(actual, expected);
    }

    private void writeToChannelAndFlush(EmbeddedChannel channel, ModelInferenceRequest msg) {
        ChannelFuture write = channel.writeAndFlush(msg);
        while (true) {
            try {
                write.sync();
                break;
            } catch (InterruptedException e) {
            }
        }
    }
}
