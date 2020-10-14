package org.pytorch.serve.plugins.ddb;

import software.amazon.awssdk.enhanced.dynamodb.extensions.annotations.DynamoDbVersionAttribute;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbBean;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbPartitionKey;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbSecondaryPartitionKey;
import software.amazon.awssdk.enhanced.dynamodb.mapper.annotations.DynamoDbSortKey;

@DynamoDbBean
public class Snapshots {
    private String snapshotName;
    private String createdOnMonth;
    private String createdOn;
    private String snapshot;
    private Long version;

    public Snapshots() {}

    @DynamoDbPartitionKey
    String getSnapshotName() {
        return snapshotName;
    }

    void setSnapshotName(String snapshotName) {
        this.snapshotName = snapshotName;
    }

    String getSnapshot() {
        return snapshot;
    }

    void setSnapshot(String snapshot) {
        this.snapshot = snapshot;
    }

    @DynamoDbSortKey
    // Defines an LSI (customers_by_date) with a sort key of 'createdDate' and also declares the
    // same attribute as a sort key for the GSI named 'customers_by_name'
    // @DynamoDbSecondarySortKey(indexNames = {"createdOnSort-index", "createdOnMonth-index"})
    String getCreatedOn() {
        return createdOn;
    }

    void setCreatedOn(String createdOn) {
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
