package topk

import (
	"container/heap"
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"io"
	"math"
	"math/rand/v2"
	"sort"
)

// HeavyKeeper implements the Top-K algorithm described in "HeavyKeeper: An
// Accurate Algorithm for Finding Top-k Elephant Flows" at
// https://www.usenix.org/system/files/conference/atc18/atc18-gong.pdf
//
// HeavyKeeper is not safe for concurrent use.
type HeavyKeeper struct {
	decay     float64
	rand      *rand.Rand
	fpSeed    maphash.Seed
	slotSeeds []maphash.Seed
	buckets   [][]bucket
	heap      minHeap
}

type bucket struct {
	fingerprint uint32
	count       uint32
}

// New returns a HeavyKeeper that tracks the k largest flows. Decay determines
// the chance that a collision will cause the existing flow count to decay. A
// decay of 0.9 is a good starting point.
//
// Width is `k * log(k)` (minimum of 256) and depth is `log(k)` (minimum of 3).
func New(k int, decay float64) *HeavyKeeper {
	if k < 1 {
		panic("k must be >= 1")
	}

	if decay <= 0 || decay > 1 {
		panic("decay must be in range (0, 1.0]")
	}

	width := int(float64(k) * math.Log(float64(k)))
	if width < 256 {
		width = 256
	}

	depth := int(math.Log(float64(k)))
	if depth < 3 {
		depth = 3
	}

	buckets := make([][]bucket, depth)
	for i := range buckets {
		buckets[i] = make([]bucket, width)
	}

	slotSeeds := make([]maphash.Seed, depth)
	for i := range slotSeeds {
		slotSeeds[i] = maphash.MakeSeed()
	}

	hk := &HeavyKeeper{
		decay:     decay,
		rand:      rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
		fpSeed:    maphash.MakeSeed(),
		slotSeeds: slotSeeds,
		buckets:   buckets,
		heap:      make(minHeap, k),
	}
	heap.Init(&hk.heap)
	return hk
}

// Sample increments the given flow's count by the given amount. It returns
// true if the flow is in the top K elements.
func (hk *HeavyKeeper) Sample(flow string, incr uint32) bool {
	fp := uint32(maphash.String(hk.fpSeed, flow))
	var maxCount uint32
	heapMin := hk.heap.Min()

	for i, row := range hk.buckets {
		j := maphash.String(hk.slotSeeds[i], flow) % uint64(len(row))

		if row[j].count == 0 {
			row[j].fingerprint = fp
			row[j].count = incr
			maxCount = max(maxCount, incr)
		} else if row[j].fingerprint == fp {
			row[j].count += incr
			maxCount = max(maxCount, row[j].count)
		} else {
			for localIncr := incr; localIncr > 0; localIncr-- {
				if hk.rand.Float64() < math.Pow(hk.decay, float64(row[j].count)) {
					row[j].count--
					if row[j].count == 0 {
						row[j].fingerprint = fp
						row[j].count = localIncr
						maxCount = max(maxCount, localIncr)
						break
					}
				}
			}
		}
	}

	if maxCount >= heapMin {
		i := hk.heap.Find(flow)
		if i > -1 {
			// update in-place if in minHeap
			hk.heap[i].Count = maxCount
			heap.Fix(&hk.heap, i)
		} else {
			hk.heap[0].Flow = flow
			hk.heap[0].Count = maxCount
			heap.Fix(&hk.heap, 0)
		}
		return true
	}

	return false
}

// FlowCount is a tuple of flow and estimated count.
type FlowCount struct {
	Flow  string
	Count uint32
}

type byCount []FlowCount

func (a byCount) Len() int           { return len(a) }
func (a byCount) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byCount) Less(i, j int) bool { return a[i].Count < a[j].Count }

func (hk *HeavyKeeper) Top() []FlowCount {
	top := make([]FlowCount, len(hk.heap))
	copy(top, hk.heap)
	sort.Stable(sort.Reverse(byCount(top)))

	// Trim off empty values
	end := len(top)
	for ; end > 0; end-- {
		if top[end-1].Count > 0 {
			break
		}
	}

	return top[:end]
}

// Count returns the estimated count of the given flow if it is in the top K
// flows.
func (hk *HeavyKeeper) Count(flow string) (count uint32, ok bool) {
	for _, hb := range hk.heap {
		if hb.Flow == flow {
			return hb.Count, true
		}
	}
	return 0, false
}

// DecayAll decays all flows by the given percentage.
func (hk *HeavyKeeper) DecayAll(pct float64) {
	if pct <= 0 {
		return
	} else if pct > 1 {
		hk.Reset()
		return
	}

	pct = 1 - pct

	for _, row := range hk.buckets {
		for i := range row {
			row[i].count = uint32(float64(row[i].count) * pct)
		}
	}
	for i := range hk.heap {
		hk.heap[i].Count = uint32(float64(hk.heap[i].Count) * pct)
	}
}

// Reset returns the HeavyKeeper to a like-new state with no flows and no
// counts.
func (hk *HeavyKeeper) Reset() {
	for _, row := range hk.buckets {
		for i := range row {
			row[i] = bucket{}
		}
	}
	for i := range hk.heap {
		hk.heap[i] = FlowCount{}
	}
}

const binaryVersion uint32 = 1

// WriteTo writes the HeavyKeeper state to w in a binary format. The top-K heap
// and structural parameters are preserved; the internal bucket sketch is not,
// since it depends on instance-specific hash seeds. After reading back with
// ReadFrom, Top() and Count() return the same results, but the bucket sketch
// starts fresh (brief warm-up period for new Sample calls).
func (hk *HeavyKeeper) WriteTo(w io.Writer) (int64, error) {
	cw := &countingWriter{w: w}

	if err := binary.Write(cw, binary.LittleEndian, binaryVersion); err != nil {
		return cw.n, fmt.Errorf("topk: write version: %w", err)
	}
	if err := binary.Write(cw, binary.LittleEndian, hk.decay); err != nil {
		return cw.n, fmt.Errorf("topk: write decay: %w", err)
	}
	if err := binary.Write(cw, binary.LittleEndian, uint32(len(hk.heap))); err != nil {
		return cw.n, fmt.Errorf("topk: write k: %w", err)
	}
	if err := binary.Write(cw, binary.LittleEndian, uint32(len(hk.buckets))); err != nil {
		return cw.n, fmt.Errorf("topk: write depth: %w", err)
	}
	if err := binary.Write(cw, binary.LittleEndian, uint32(len(hk.buckets[0]))); err != nil {
		return cw.n, fmt.Errorf("topk: write width: %w", err)
	}

	for _, fc := range hk.heap {
		if err := binary.Write(cw, binary.LittleEndian, uint32(len(fc.Flow))); err != nil {
			return cw.n, fmt.Errorf("topk: write flow length: %w", err)
		}
		if _, err := cw.Write([]byte(fc.Flow)); err != nil {
			return cw.n, fmt.Errorf("topk: write flow: %w", err)
		}
		if err := binary.Write(cw, binary.LittleEndian, fc.Count); err != nil {
			return cw.n, fmt.Errorf("topk: write count: %w", err)
		}
	}

	return cw.n, nil
}

// ReadFrom reads HeavyKeeper state from r, replacing the current state. The
// internal bucket sketch is reset with new hash seeds; see WriteTo for details.
func (hk *HeavyKeeper) ReadFrom(r io.Reader) (int64, error) {
	cr := &countingReader{r: r}

	var version uint32
	if err := binary.Read(cr, binary.LittleEndian, &version); err != nil {
		return cr.n, fmt.Errorf("topk: read version: %w", err)
	}
	if version != binaryVersion {
		return cr.n, fmt.Errorf("topk: unsupported version %d", version)
	}

	var decay float64
	if err := binary.Read(cr, binary.LittleEndian, &decay); err != nil {
		return cr.n, fmt.Errorf("topk: read decay: %w", err)
	}

	var k, depth, width uint32
	if err := binary.Read(cr, binary.LittleEndian, &k); err != nil {
		return cr.n, fmt.Errorf("topk: read k: %w", err)
	}
	if err := binary.Read(cr, binary.LittleEndian, &depth); err != nil {
		return cr.n, fmt.Errorf("topk: read depth: %w", err)
	}
	if err := binary.Read(cr, binary.LittleEndian, &width); err != nil {
		return cr.n, fmt.Errorf("topk: read width: %w", err)
	}

	h := make(minHeap, k)
	for i := range h {
		var flowLen uint32
		if err := binary.Read(cr, binary.LittleEndian, &flowLen); err != nil {
			return cr.n, fmt.Errorf("topk: read flow length: %w", err)
		}
		flowBuf := make([]byte, flowLen)
		if _, err := io.ReadFull(cr, flowBuf); err != nil {
			return cr.n, fmt.Errorf("topk: read flow: %w", err)
		}
		if err := binary.Read(cr, binary.LittleEndian, &h[i].Count); err != nil {
			return cr.n, fmt.Errorf("topk: read count: %w", err)
		}
		h[i].Flow = string(flowBuf)
	}

	buckets := make([][]bucket, depth)
	for i := range buckets {
		buckets[i] = make([]bucket, width)
	}

	slotSeeds := make([]maphash.Seed, depth)
	for i := range slotSeeds {
		slotSeeds[i] = maphash.MakeSeed()
	}

	hk.decay = decay
	hk.rand = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	hk.fpSeed = maphash.MakeSeed()
	hk.slotSeeds = slotSeeds
	hk.buckets = buckets
	hk.heap = h
	heap.Init(&hk.heap)

	return cr.n, nil
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	n, err := cw.w.Write(p)
	cw.n += int64(n)
	return n, err
}

type countingReader struct {
	r io.Reader
	n int64
}

func (cr *countingReader) Read(p []byte) (int, error) {
	n, err := cr.r.Read(p)
	cr.n += int64(n)
	return n, err
}

type minHeap []FlowCount

var _ heap.Interface = &minHeap{}

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].Count < h[j].Count }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any) { *h = append(*h, x.(FlowCount)) }

func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// Min returns the minimum count in the heap or 0 if the heap is empty.
func (h minHeap) Min() uint32 {
	return h[0].Count
}

// Find returns the index of the given flow in the heap so that it can be
// updated in-place (be sure to call heap.Fix() afterwards). It returns -1 if
// the flow doesn't exist in the heap.
func (h minHeap) Find(flow string) (i int) {
	for i := range h {
		if h[i].Flow == flow {
			return i
		}
	}
	return -1
}
