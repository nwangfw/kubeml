package main

import (
	"fmt"
	"net/url"

	"github.com/nwangfw/kubeml/ml/pkg/api"
)

func main() {
	u, _ := url.Parse(api.StorageUrl + "/dataset/test")
	fmt.Println(u.Host, u.Scheme, u.User, u.Path)

}
