package org.pytorch.serve.plugins.ddb.snapshot;

import software.amazon.awssdk.enhanced.dynamodb.extensions.annotations.DynamoDbVersionAttribute;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbBean;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbPartitionKey;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbSecondaryPartitionKey;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbSortKey;

@DynamoDbBean
public class DDBSnapshot {
    private String snapshotName;
    private String createdOnMonth;
    private String createdOn;
    private String snapshot;
    private Long version;

    public DDBSnapshot() {}

    @DynamoDbPartitionKey
    public String getSnapshotName() {
        return snapshotName;
    }

    public void setSnapshotName(String snapshotName) {
        this.snapshotName = snapshotName;
    }

    public String getSnapshot() {
        return snapshot;
    }

    public void setSnapshot(String snapshot) {
        this.snapshot = snapshot;
    }

    @DynamoDbSortKey
    public String getCreatedOn() {
        return createdOn;
    }

    public void setCreatedOn(String createdOn) {
        this.createdOn = createdOn;
    }

    @DynamoDbVersionAttribute
    public Long getVersion() {
        return version;
    }

    public void setVersion(Long version) {
        this.version = version;
    }

    @DynamoDbSecondaryPartitionKey(indexNames = {"createdOnMonth-index"})
    public String getCreatedOnMonth() {
        return createdOnMonth;
    }

    public void setCreatedOnMonth(String createdOnMonth) {
        this.createdOnMonth = createdOnMonth;
    }
}
