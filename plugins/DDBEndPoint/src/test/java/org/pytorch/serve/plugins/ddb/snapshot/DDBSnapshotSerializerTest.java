package org.pytorch.serve.plugins.ddb.snapshot;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Properties;
import org.pytorch.serve.plugins.ddb.snapshot.ddb.DDBClient;
import org.pytorch.serve.servingsdk.snapshot.Snapshot;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbEnhancedClient;
import software.amazon.awssdk.enhanced.dynamodb.DynamoDbTable;
import software.amazon.awssdk.enhanced.dynamodb.model.CreateTableEnhancedRequest;
import software.amazon.awssdk.enhanced.dynamodb.model.EnhancedGlobalSecondaryIndex;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.DeleteTableRequest;
import software.amazon.awssdk.services.dynamodb.model.DescribeTableRequest;
import software.amazon.awssdk.services.dynamodb.model.DescribeTableResponse;
import software.amazon.awssdk.services.dynamodb.model.Projection;
import software.amazon.awssdk.services.dynamodb.model.ProjectionType;
import software.amazon.awssdk.services.dynamodb.model.ProvisionedThroughput;

public class DDBSnapshotSerializerTest {
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private DynamoDbClient dynamoDbClient;
    private DynamoDbEnhancedClient dynamodbClientEnhanced;
    private LocalDynamoDB server;
    private DynamoDbTable<DDBSnapshot> snapshotTable;

    @BeforeSuite
    void init() {
        server = new LocalDynamoDB();
        server.start(); // Start the service running locally on host
        dynamoDbClient = server.createClient();
        dynamodbClientEnhanced =
                DynamoDbEnhancedClient.builder().dynamoDbClient(dynamoDbClient).build();
        snapshotTable =
                dynamodbClientEnhanced.table(
                        DDBSnapshotSerializer.TABLE_NAME, DDBSnapshotSerializer.getTableSchema());
        createTable();
    }

    @AfterSuite
    void stop() {
        deleteTable();
        server.stop();
    }

    private void createTable() {
        ProvisionedThroughput provThroughput =
                ProvisionedThroughput.builder()
                        .readCapacityUnits(10L)
                        .writeCapacityUnits(5L)
                        .build();
        Projection projectAll = Projection.builder().projectionType(ProjectionType.ALL).build();
        EnhancedGlobalSecondaryIndex gsi =
                EnhancedGlobalSecondaryIndex.builder()
                        .indexName("createdOnMonth-index")
                        .projection(projectAll)
                        .provisionedThroughput(provThroughput)
                        .build();

        CreateTableEnhancedRequest createTableReq =
                CreateTableEnhancedRequest.builder()
                        .globalSecondaryIndices(Arrays.asList(gsi))
                        .build();
        snapshotTable.createTable(createTableReq);
    }

    public void deleteTable() {
        dynamoDbClient.deleteTable(
                DeleteTableRequest.builder().tableName(DDBSnapshotSerializer.TABLE_NAME).build());
    }

    @Test
    public void testValidateSnapshotTable() {
        DescribeTableResponse resp =
                dynamoDbClient.describeTable(
                        DescribeTableRequest.builder()
                                .tableName(DDBSnapshotSerializer.TABLE_NAME)
                                .build());
        Assert.assertEquals(resp.table().tableStatusAsString(), "ACTIVE");
    }

    @Test
    public void testSaveDDBSnapshot() throws IOException {

        DDBClient.initInstance(dynamodbClientEnhanced);
        DDBSnapshotSerializer ddbSnapSerializer = new DDBSnapshotSerializer();
        Snapshot snap = saveSnapShot("Test", ddbSnapSerializer);
        // Validate saved snapshot
        DDBSnapshot itemResponse = snapshotTable.getItem(getDDBSnapshot(snap));
        Assert.assertEquals(itemResponse.getSnapshotName(), getPartKey(snap));
    }

    @Test
    public void testGetLastDDBSnapshot() throws IOException {
        // Add snapshots
        DDBClient.initInstance(dynamodbClientEnhanced);
        DDBSnapshotSerializer ddbSnapSerializer = new DDBSnapshotSerializer();
        saveSnapShot("Test1", ddbSnapSerializer);
        saveSnapShot("Test2", ddbSnapSerializer);
        saveSnapShot("Test3", ddbSnapSerializer);
        saveSnapShot("Test4", ddbSnapSerializer);

        Properties snapshot = ddbSnapSerializer.getLastSnapshot();
        Assert.assertEquals(snapshot.get("batch_delay"), "100");
    }

    public Snapshot saveSnapShot(String name, DDBSnapshotSerializer ddbSnapSerializer)
            throws IOException {
        Snapshot snap = getSnapshot(name);
        ddbSnapSerializer.saveSnapshot(snap, getConfig());
        return snap;
    }

    public Properties getConfig() {
        Properties config = new Properties();
        config.setProperty("batch_size", "5");
        config.setProperty("batch_delay", "100");
        return config;
    }

    public Snapshot getSnapshot(String name) {
        Snapshot snap = new Snapshot(name, 0);
        long now = System.currentTimeMillis();
        snap.setCreated(now);
        return snap;
    }

    public String getPartKey(Snapshot snap) {
        return snap.getName() + "_" + snap.getCreated() + "_" + snap.getModelCount();
    }

    public String getCreatedOnMonth(long createdOn) {
        Date createdOnDt = new Date(createdOn);
        SimpleDateFormat monthFmt = new SimpleDateFormat("yyyy-MM");
        return monthFmt.format(createdOnDt);
    }

    public String getCreatedOn(long createdOn) {
        Date createdOnDt = new Date(createdOn);
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS");
        return sdf.format(createdOnDt);
    }

    public Properties getConfigWithSnap(Snapshot snap) {
        String snapshotJson = GSON.toJson(snap, Snapshot.class);
        Properties config = getConfig();
        config.put("model_snapshot", snapshotJson);
        return config;
    }

    public DDBSnapshot getDDBSnapshot(Snapshot snap) throws IOException {
        DDBSnapshot ddbSnapshot = new DDBSnapshot();
        ddbSnapshot.setSnapshotName(getPartKey(snap));
        ddbSnapshot.setCreatedOn(getCreatedOn(snap.getCreated()));
        ddbSnapshot.setCreatedOnMonth(getCreatedOnMonth(snap.getCreated()));
        ddbSnapshot.setSnapshot(DDBSnapshotSerializer.convert2String(getConfigWithSnap(snap)));
        return ddbSnapshot;
    }
}
