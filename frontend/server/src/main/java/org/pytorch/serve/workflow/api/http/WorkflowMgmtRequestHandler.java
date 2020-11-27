package org.pytorch.serve.workflow.api.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.ensemble.WorkFlow;
import org.pytorch.serve.http.*;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.util.NettyUtils;
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

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException {
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
                        && segments[1].equals("workflows"))
                || endpointMap.containsKey(segments[1]);
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

    private void handleDescribeWorkflow(ChannelHandlerContext ctx, String workflowName) {
        ArrayList<DescribeWorkflowResponse> resp = new ArrayList<>();
        WorkFlow workFlow = WorkflowManager.getInstance().getWorkflow(workflowName);
        resp.add(createWorkflowResponse(workflowName, workFlow));
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleRegisterWorkflows(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req) {
        StatusResponse status = new StatusResponse();

        try {
            RegisterWorkflowRequest registerWFRequest = parseRequest(req, decoder);

            status =
                    WorkflowManager.getInstance()
                            .registerWorkflow(
                                    registerWFRequest.getWorkflowName(),
                                    registerWFRequest.getWorkflowUrl(),
                                    registerWFRequest.getResponseTimeout(),
                                    true);

        } catch (InterruptedException | ExecutionException | IOException | ConflictStatusException e) {
            status.setHttpResponseCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
            status.setStatus("Error while registering workflow. Details: " + e.getMessage());
            status.setE(e);

        } catch (Exception e) {
            status.setHttpResponseCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
            status.setStatus("Error while registering workflow. Details: " + e.getMessage());
            status.setE(e);

        } finally {
            sendResponse(ctx, status);
        }
    }

    private void handleUnregisterWorkflow(ChannelHandlerContext ctx, String workflowName) {
        WorkflowManager.getInstance().unregisterWorkflow(workflowName);
        String msg = "Workflow \"" + workflowName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg, HttpResponseStatus.OK.code()));
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
            if (statusResponse.getHttpResponseCode() >= 200
                    && statusResponse.getHttpResponseCode() < 300) {
                NettyUtils.sendJsonResponse(ctx, statusResponse);
            } else {
                // Re-map HTTPURLConnections HTTP_ENTITY_TOO_LARGE to Netty's INSUFFICIENT_STORAGE
                int httpResponseStatus = statusResponse.getHttpResponseCode();
                NettyUtils.sendError(
                        ctx,
                        HttpResponseStatus.valueOf(
                                httpResponseStatus == 413 ? 507 : httpResponseStatus),
                        statusResponse.getE());
            }
        }
    }

    private static DescribeWorkflowResponse createWorkflowResponse(
            String workflowName, WorkFlow workflow) {
        DescribeWorkflowResponse response = new DescribeWorkflowResponse();
        response.setWorkflowName(workflowName);
        response.setWorkflowUrl(workflow.getWorkflowArchive().getUrl());
        response.setBatchSize(workflow.getBatchSize());
        response.setMaxBatchDelay(workflow.getBatchSizeDelay());
        response.setMaxWorkers(workflow.getMaxWorkers());
        response.setMinWorkers(workflow.getMinWorkers());
        response.setWorkflowDag(workflow.getWorkflowDag());
        return response;
    }
}
