package org.pytorch.serve;

import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.security.GeneralSecurityException;
import  org.pytorch.serve.ensemble.Dag;

public class EnsembleTest {

    static {
        TestUtils.init();
    }


    @BeforeClass
    public void beforeSuite()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {

    }

    @AfterClass
    public void afterSuite() throws InterruptedException {
        TestUtils.closeChannels();
    }

    @Test
    public void testStartupSnapshot() {
       Dag dag= new Dag();
       Node a = new Node<>("a");
       Node b = new Node<>("b");
       Node c = new Node<>("c");
       Node d = new Node<>("d");
       Node e = new Node<>("e");
        Node f = new Node<>("f");


       dag.addNode(a);
       dag.addNode(b);
       dag.addNode(c);
       dag.addNode(d);
       dag.addNode(e);
       dag.addNode(f);

        try {
            dag.addEdge(a, b);
            dag.addEdge(a, c);
            dag.addEdge(a, d);
            dag.addEdge(b, e);
            dag.addEdge(e, f);
            dag.addEdge(c, f);
            dag.addEdge(d, f);
        } catch (Exception exp) {
            exp.printStackTrace();
        }

        try {
            System.out.println(dag.topoSort());
        } catch (Exception exp) {
            exp.printStackTrace();
        }
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testStartupSnapshot"})
    public void testUnregisterSnapshot() throws InterruptedException {
        assert 1==1;

    }


}
