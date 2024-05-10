package org.pytorch.serve.job;

import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class JobGroup {
    private static final Logger logger = LoggerFactory.getLogger(JobGroup.class);
    String groupId;
    LinkedBlockingDeque<Job> jobs;
    int maxJobQueueSize;
    boolean groupEnd;

    public JobGroup(String groupId, int maxJobQueueSize) {
        this.groupId = groupId;
        this.maxJobQueueSize = maxJobQueueSize;
        this.jobs = new LinkedBlockingDeque<>(maxJobQueueSize);
        this.groupEnd = false;
    }

    public boolean appendJob(Job job) {
        return jobs.offer(job);
    }

    public Job pollJob(long timeout) {
        if (groupEnd) {
            return null;
        }
        try {
            return jobs.poll(timeout, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            logger.error("Failed to poll a job from group {}", groupId, e);
        }
        return null;
    }

    public String getGroupId() {
        return groupId;
    }

    public void setGroupEnd(boolean sequenceEnd) {
        this.groupEnd = sequenceEnd;
    }

    public boolean isGroupEnd() {
        return this.groupEnd;
    }
}
