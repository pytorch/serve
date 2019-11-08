package org.pytorch.serve.archive;

import java.io.IOException;
import org.pytorch.serve.test.TestHelper;
import org.testng.annotations.Test;

public class CoverageTest {

    @Test
    public void test() throws IOException, ClassNotFoundException {
        TestHelper.testGetterSetters(ModelArchive.class);
    }
}
