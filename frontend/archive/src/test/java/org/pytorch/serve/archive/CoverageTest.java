package org.pytorch.serve.archive;

import java.io.IOException;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.pytorch.serve.test.TestHelper;
import org.testng.annotations.Test;

public class CoverageTest {

    @Test
    public void testModelArchive() throws IOException, ClassNotFoundException {
        TestHelper.testGetterSetters(ModelArchive.class);
    }

    @Test
    public void testWorkflowArchive() throws IOException, ClassNotFoundException {
        TestHelper.testGetterSetters(WorkflowArchive.class);
    }
}
