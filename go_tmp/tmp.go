package main

import (
	"fmt"
	"log"
	"os"

	npyio "github.com/sbinet/npyio"
)

func main() {
	f, _ := os.Open("/data/cwj/repgnn/ogbn_arxiv600/feat.npy")
	defer f.Close()
	// err = npyio.Read(f, &data)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Printf("data = %v %v\n", data[:3], len(data))

	r, err := npyio.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("npy-header: %v\n", r.Header)
	shape := r.Header.Descr.Shape
	raw := make([]float32, shape[0]*shape[1])

	err = r.Read(&raw)
	if err != nil {
		log.Fatal(err)
	}

	// m := mat.NewDense(shape[0], shape[1], raw)
	// fmt.Printf("data = %v\n", mat.Formatted(m, mat.Prefix("       ")))
	fmt.Printf("%v", raw[0])

}
