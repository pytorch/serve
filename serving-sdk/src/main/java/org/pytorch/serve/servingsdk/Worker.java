

package org.pytorch.serve.servingsdk;

/**
 * Describe the model worker
 */
public interface Worker {
    /**
     * Get the current running status of this model's worker
     * @return True - if the worker is currently running. False - the worker is currently not running.
     */
    boolean isRunning();

    /**
     * Get the current memory foot print of this worker
     * @return Current memory usage of this worker
     */
    long getWorkerMemory();
}
