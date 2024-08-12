package org.pytorch.serve.workflow.api.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import java.net.HttpURLConnection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowNotFoundException;
import org.pytorch.serve.ensemble.WorkFlow;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.MethodNotAllowedException;
import org.pytorch.serve.http.ResourceNotFoundException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.pytorch.serve.workflow.WorkflowManager;
import org.pytorch.serve.workflow.messages.DescribeWorkflowResponse;
import org.pytorch.serve.workflow.messages.ListWorkflowResponse;
import org.pytorch.serve.workflow.messages.RegisterWorkflowRequest;

/**
 * A class handling inbound HTTP requests to the workflow management API.
 *
 * <p>This class
 */
public class WorkflowMgmtRequestHandler extends HttpRequestHandlerChain {

    /** Creates a new {@code WorkflowMgmtRequestHandler} instance. */
    public WorkflowMgmtRequestHandler() {}

    private static DescribeWorkflowResponse createWorkflowResponse(
            String workflowName, WorkFlow workflow) {
        DescribeWorkflowResponse response = new DescribeWorkflowResponse();
        response.setWorkflowName(workflowName);
        response.setWorkflowUrl(workflow.getWorkflowArchive().getUrl());
        response.setBatchSize(workflow.getBatchSize());
        response.setMaxBatchDelay(workflow.getMaxBatchDelay());
        response.setMaxWorkers(workflow.getMaxWorkers());
        response.setMinWorkers(workflow.getMinWorkers());
        response.setWorkflowDag(workflow.getWorkflowDag());
        return response;
    }

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {

        if (isManagementReq(segments)) {
            if (!"workflows".equals(segments[1])) {
                throw new ResourceNotFoundException();
            }

            HttpMethod method = req.method();
            if (segments.length < 3) {
                if (HttpMethod.GET.equals(method)) {
                    handleListWorkflows(ctx, decoder);
                    return;
                } else if (HttpMethod.POST.equals(method)) {
                    handleRegisterWorkflows(ctx, decoder, req);
                    return;
                }
                throw new MethodNotAllowedException();
            }

            if (HttpMethod.GET.equals(method)) {
                handleDescribeWorkflow(ctx, segments[2]);
            } else if (HttpMethod.DELETE.equals(method)) {
                handleUnregisterWorkflow(ctx, segments[2]);
            } else {
                throw new MethodNotAllowedException();
            }
        } else {
            chain.handleRequest(ctx, req, decoder, segments);
        }
    }

    private boolean isManagementReq(String[] segments) {
        return segments.length == 0
                || ((segments.length >= 2 && segments.length <= 4)
                        && segments[1].equals("workflows"));
    }

    private void handleListWorkflows(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
        int pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);
        if (limit > 100 || limit < 0) {
            limit = 100;
        }
        if (pageToken < 0) {
            pageToken = 0;
        }

        Map<String, WorkFlow> workflows = WorkflowManager.getInstance().getWorkflows();

        List<String> keys = new ArrayList<>(workflows.keySet());
        Collections.sort(keys);
        ListWorkflowResponse list = new ListWorkflowResponse();

        int last = pageToken + limit;
        if (last > keys.size()) {
            last = keys.size();
        } else {
            list.setNextPageToken(String.valueOf(last));
        }

        for (int i = pageToken; i < last; ++i) {
            String workflowName = keys.get(i);
            WorkFlow workFlow = workflows.get(workflowName);
            list.addModel(workflowName, workFlow.getWorkflowArchive().getUrl());
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleDescribeWorkflow(ChannelHandlerContext ctx, String workflowName)
            throws WorkflowNotFoundException {
        ArrayList<DescribeWorkflowResponse> resp = new ArrayList<>();
        WorkFlow workFlow = WorkflowManager.getInstance().getWorkflow(workflowName);
        if (workFlow == null) {
            throw new WorkflowNotFoundException("Workflow not found: " + workflowName);
        }
        resp.add(createWorkflowResponse(workflowName, workFlow));
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleRegisterWorkflows(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req)
            throws ConflictStatusException, WorkflowException {
        RegisterWorkflowRequest registerWFRequest = parseRequest(req, decoder);

        StatusResponse status =
                WorkflowManager.getInstance()
                        .registerWorkflow(
                                registerWFRequest.getWorkflowName(),
                                registerWFRequest.getWorkflowUrl(),
                                registerWFRequest.getResponseTimeout(),
                                registerWFRequest.getStartupTimeout(),
                                true,
                                registerWFRequest.getS3SseKms());

        sendResponse(ctx, status);
    }

    private void handleUnregisterWorkflow(ChannelHandlerContext ctx, String workflowName)
            throws WorkflowNotFoundException {
        StatusResponse statusResponse = null;
        try {
            WorkflowManager.getInstance().unregisterWorkflow(workflowName, null);
            String msg = "Workflow \"" + workflowName + "\" unregistered";
            statusResponse = new StatusResponse(msg, HttpResponseStatus.OK.code());
        } catch (InterruptedException | ExecutionException e) {
            String msg =
                    "Error while unregistering the workflow "
                            + workflowName
                            + ". Workflow not found.";
            statusResponse =
                    new StatusResponse(msg, HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        }

        NettyUtils.sendJsonResponse(ctx, statusResponse);
    }

    private RegisterWorkflowRequest parseRequest(FullHttpRequest req, QueryStringDecoder decoder) {
        RegisterWorkflowRequest in;
        CharSequence mime = HttpUtil.getMimeType(req);
        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(mime)) {
            in =
                    JsonUtils.GSON.fromJson(
                            req.content().toString(CharsetUtil.UTF_8),
                            RegisterWorkflowRequest.class);
        } else {
            in = new RegisterWorkflowRequest(decoder);
        }
        return in;
    }

    private void sendResponse(ChannelHandlerContext ctx, StatusResponse statusResponse) {
        if (statusResponse != null) {
            if (statusResponse.getHttpResponseCode() >= HttpURLConnection.HTTP_OK
                    && statusResponse.getHttpResponseCode() < HttpURLConnection.HTTP_MULT_CHOICE) {
                NettyUtils.sendJsonResponse(ctx, statusResponse);
            } else {
                // Re-map HTTPURLConnections HTTP_ENTITY_TOO_LARGE to Netty's INSUFFICIENT_STORAGE
                int httpResponseStatus = statusResponse.getHttpResponseCode();
                NettyUtils.sendError(
                        ctx,
                        HttpResponseStatus.valueOf(
                                httpResponseStatus == HttpURLConnection.HTTP_ENTITY_TOO_LARGE
                                        ? 507
                                        : httpResponseStatus),
                        statusResponse.getE());
            }
        }
    }
}
