package org.pytorch.serve.util.codec;

import com.google.common.primitives.Bytes;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandler;
import io.netty.channel.embedded.EmbeddedChannel;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.RequestInput;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class ModelRequestEncoderTest {

    @DataProvider(name = "largeCompressed")
    private Object[] createLargeCompressed() {
        URL url = this.getClass().getResource("/large.zst");
        Path path;
        try {
            path = Path.of(url.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        try {
            return new Object[] {Files.readAllBytes(path)};
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

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

        input.addParameter(
                new InputParameter("input_name", compressed, "application/json", new String[0]));
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

    @Test
    public void testZstdSimple() throws IOException {
        ChannelHandler encoder = new ModelRequestEncoder(true);
        EmbeddedChannel channel = new EmbeddedChannel(encoder);
        ModelInferenceRequest msg = new ModelInferenceRequest("testModel");
        ArrayList list = new ArrayList<RequestInput>();
        RequestInput input = new RequestInput("request_id");
        byte[] compressed =
                new byte[] {
                    0x28,
                    (byte) 0xB5,
                    0x2F,
                    (byte) 0xFD,
                    0x04,
                    0x58,
                    (byte) 0x91,
                    0x00,
                    0x00,
                    0x7B,
                    0x22,
                    0x64,
                    0x61,
                    0x74,
                    0x61,
                    0x22,
                    0x3A,
                    0x20,
                    0x22,
                    0x76,
                    0x61,
                    0x6C,
                    0x75,
                    0x65,
                    0x22,
                    0x7D,
                    0x0A,
                    (byte) 0x87,
                    (byte) 0xED,
                    (byte) 0x94,
                    0x07
                };

        input.addParameter(
                new InputParameter(
                        "input_name", compressed, "application/json", new String[] {"zstd"}));
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
                    (byte) 0x12,
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
                    '\n',
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

    @Test(dataProvider = "largeCompressed")
    public void testZstdLarge(byte[] largeCompressed) throws IOException {
        ChannelHandler encoder = new ModelRequestEncoder(true);
        EmbeddedChannel channel = new EmbeddedChannel(encoder);
        ModelInferenceRequest msg = new ModelInferenceRequest("testModel");
        ArrayList list = new ArrayList<RequestInput>();
        RequestInput input = new RequestInput("request_id");

        input.addParameter(
                new InputParameter(
                        "input_name", largeCompressed, "plain/text", new String[] {"zstd"}));
        list.add(input);
        msg.setRequestBatch(list);
        writeToChannelAndFlush(channel, msg);

        byte[] expectedStart =
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
                    0x0A,
                    'p',
                    'l',
                    'a',
                    'i',
                    'n',
                    '/',
                    't',
                    'e',
                    'x',
                    't',
                    0x00,
                    (byte) 0x9C,
                    0x40,
                    0x01,
                };

        byte[] expectedValue = new byte[10240001];
        for (int i = 0; i != expectedValue.length; i++) {
            expectedValue[i] = '-';
        }

        expectedValue[expectedValue.length - 1] = '\n';

        byte[] expectedEnd =
                new byte[] {
                    (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, // end of parameters
                    (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF // end of batch
                };

        byte[] expected = Bytes.concat(expectedStart, expectedValue, expectedEnd);
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
        Assert.assertEquals(actual, expected);
    }

    private void writeToChannelAndFlush(EmbeddedChannel channel, ModelInferenceRequest msg) {
        ChannelFuture write = channel.writeAndFlush(msg);
        while (true) {
            try {
                write.sync();
                break;
            } catch (InterruptedException e) {
                continue;
            }
        }
    }
}
