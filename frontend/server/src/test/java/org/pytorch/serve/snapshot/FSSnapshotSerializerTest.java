package org.pytorch.serve.snapshot;

import org.testng.annotations.Test;

import static org.testng.Assert.*;

public class FSSnapshotSerializerTest {

  @Test
  public void testGetSnapshotTime() {
    String filename = "123-not-in-use";
    long value = FSSnapshotSerializer.getSnapshotTime(filename);
    assertEquals(123L, value);
  }
}