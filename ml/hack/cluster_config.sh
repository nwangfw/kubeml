#!/bin/bash

#
# cluster_config.sh - Automate setup of cluster
#


# declare the namespaces
FISSION_NAMESPACE="fission"
MONITORING_NAMESPACE="monitoring"
KUBEML_NAMESPACE="kubeml"

# check if kubectl is installed
if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl is not installed"
    exit 1;
fi

# Check if helm is installed
if ! command -v helm >/dev/null 2>&1 ; then
    echo "helm is not installed."
    exit 1;
fi

# Create the fission release
echo "Deploying fission..."

kubectl create namespace $FISSION_NAMESPACE
helm repo add fission-charts https://fission.github.io/fission-charts/
helm repo update
helm install --version v1.17.0 --namespace $FISSION_NAMESPACE fission \
    fission-charts/fission-all --set prometheus.enabled=False \
    2>&1

echo "Fission deployed!"


# if the env variable is not set create monitoring namespace and resources
if [[ -z $MONITORING ]]; then
  echo "Deploying prometheus..."

  kubectl create namespace $MONITORING_NAMESPACE
  helm install kubeml-metrics --namespace $MONITORING_NAMESPACE prometheus-community/kube-prometheus-stack \
  --set kubelet.serviceMonitor.https=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
  2>&1

  echo "Prometheus deployed!"
fi

# Deploy the kubeml charts
echo "Deploying kubeml"

kubectl create namespace $KUBEML_NAMESPACE
helm install kubeml ./ml/charts/kubeml --namespace $KUBEML_NAMESPACE \
    2>&1

echo "kubeml deployed!! all done"

